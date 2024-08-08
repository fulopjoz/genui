"""
builders

Created by: Martin Sicho
On: 15-01-20, 12:55
"""
# CHANGE: Added import for traceback to handle exceptions
import traceback

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from rdkit import Chem

from genui.models.genuimodels.bases import Algorithm
from genui.models.models import ModelPerformanceCV
from . import bases
# CHANGE: Updated imports to use more specific model references
from genui.models import models as core_models
from genui.qsar import models as qsar_models
from sklearn.model_selection import KFold, StratifiedKFold

class BasicQSARModelBuilder(bases.QSARModelBuilder):
    # CHANGE: Updated __init__ method to handle multiple validation strategies
    def __init__(self, instance: qsar_models.Model, progress=None, onFitCall=None, validations=None):
        super().__init__(instance, progress, onFitCall)
        # Use provided validations or fetch all from the training strategy
        self.validations = validations if validations and len(validations) \
            > 0 else self.instance.trainingStrategy.validationStrategies.all()
            # Checks 
        # CHANGE: Allow for custom validation strategies or use all associated with the training strategy.
        # This provides flexibility in choosing validation approaches for each model build.
    def build(self) -> qsar_models.QSARModel:
        if not self.validations:
            raise ImproperlyConfigured("You cannot build a QSAR model without validation strategies.")
        # CHANGE: Now checking for the presence of validation strategies instead of a single validation strategy.
        # This ensures that at least one validation strategy is present before building the model.
        if not self.molsets:
            raise ImproperlyConfigured("You cannot build a QSAR model without an associated molecule set.")

        self.progressStages = [
            "Fetching activities...",
            "Calculating descriptors..."
        ]
        # CHANGE: Generate progress stages for each validation strategy
        for validation in self.validations:
            self.progressStages.extend([f"CV fold {x+1}" for x in range(validation.cvFolds)])
        self.progressStages.extend(["Fitting model on the training set...", "Validating on test set..."])
        self.progressStages.extend(["Fitting the final model..."])

        self.recordProgress()
        mols = self.saveActivities()[1]

        self.recordProgress()
        self.calculateDescriptors(mols)
        # CHANGE: Apply each validation strategy separately
        for validation in self.validations: 
            if hasattr(validation, 'valid_set_size') and hasattr(validation, 'cvFolds'):
                # handle multiple validation strategies
                X_valid = self.X.sample(frac=validation.valid_set_size)
                X_train = self.X.drop(X_valid.index)
                y_valid = self.y[X_valid.index]
                y_train = self.y.drop(y_valid.index)

                is_regression = self.training.mode.name == Algorithm.REGRESSION
                if is_regression:
                    folds = KFold(validation.cvFolds).split(X_train)
                else:
                    folds = StratifiedKFold(validation.cvFolds).split(X_train, y_train)
                for i, (trained, validated) in enumerate(folds):
                    self.recordProgress()
                    self.fitAndValidate(X_train.iloc[trained], y_train.iloc[trained], X_train.iloc[validated], y_train.iloc[validated], perfClass=core_models.ModelPerformanceCV, fold=i + 1)

        model = self.algorithmClass(self)
        self.recordProgress()
        model.fit(X_train, y_train)
        self.recordProgress()
        self.fitAndValidate(X_train, y_train, X_valid, y_valid)
        self.recordProgress()
        return super().build()

    def predictMolecules(self, mols):
        smiles = []
        failed_indices = []
        for idx, mol in enumerate(mols):
            if mol:
                smiles.append(Chem.MolToSmiles(mol) if type(mol) == Chem.Mol else mol)
            else:
                failed_indices.append(idx)

        predictions = [-1] * len(mols)
        if len(failed_indices) == len(mols):
            return np.array(predictions)

        self.calculateDescriptors(smiles)

        real_predictions = list(self.predict(self.getX()))
        for idx,prediction in enumerate(predictions):
            if idx not in failed_indices:
                predictions[idx] = real_predictions.pop(0)
        assert len(real_predictions) == 0
        return np.array(predictions)
    
    # CHANGE: Updated to use qsar_models instead of models
    def populateActivitySet(self, aset : qsar_models.ModelActivitySet):
        if not self.instance.predictionsType:
            raise Exception("The activity type for QSAR model predictions is not specified.")

        aset.activities.all().delete()
        molecules = aset.molecules.molecules.all()
        predictions = self.predictMolecules(molecules)

        for mol, prediction in zip(molecules, predictions):
            qsar_models.ModelActivity.objects.create(
                value=prediction,
                type=self.instance.predictionsType,
                units=self.instance.predictionsUnits,
                source=aset,
                molecule=mol,
            )

        return aset.activities.all()
    
    # CHANGE: Added new method to handle fitting and validation
    def fitAndValidate(self, X_train, y_train, X_valid, y_valid, y_predicted=None, perfClass=core_models.ModelPerformance, *args, **kwargs):
        if not y_predicted:
            model = self.algorithmClass(self)
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_valid)
        for validation in self.validations:
            self.validate(validation, y_valid, y_predicted, perfClass, *args, **kwargs)
            
    # CHANGE: Added new method to handle individual validation
    def validate(self, validation_strategy, y_validated, y_predicted, perfClass=core_models.ModelPerformance, *args, **kwargs):
        if not validation_strategy:
            raise ImproperlyConfigured(f"No validation strategy is set for model: {repr(self.instance)}")
        for metric_class in self.metricClasses:
            try:
                metric_class(self).save(y_validated, y_predicted, perfClass, *args, **kwargs)
            except Exception as exp:
                print("Failed to obtain values for metric: ", metric_class.name)
                self.errors.append(exp)
                traceback.print_exc()
                continue
            