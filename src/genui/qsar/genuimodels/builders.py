"""
builders

Created by: Martin Sicho
On: 15-01-20, 12:55
"""

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from rdkit import Chem

from genui.models.genuimodels.bases import Algorithm
from genui.models.models import ModelPerformanceCV
from . import bases
from genui.qsar import models
from sklearn.model_selection import KFold, StratifiedKFold

class BasicQSARModelBuilder(bases.QSARModelBuilder):

    def build(self) -> models.QSARModel:
        if not self.validation:
            raise ImproperlyConfigured("You cannot build a QSAR model with a missing validation strategy.")

        if not self.molsets:
            raise ImproperlyConfigured("You cannot build a QSAR model without an associated molecule set.")

        self.progressStages = [
            "Fetching activities...",
            "Calculating descriptors..."
        ]
        self.progressStages.extend([f"CV fold {x+1}" for x in range(self.validation.cvFolds)])
        self.progressStages.extend(["Fitting model on the training set...", "Validating on test set..."])
        self.progressStages.extend(["Fitting the final model..."])

        self.recordProgress()
        mols = self.saveActivities()[1]

        self.recordProgress()
        self.calculateDescriptors(mols)

        X_valid = self.X.sample(frac=self.validation.validSetSize)
        X_train = self.X.drop(X_valid.index)
        y_valid = self.y[X_valid.index]
        y_train = self.y.drop(y_valid.index)

        is_regression = self.training.mode.name == Algorithm.REGRESSION
        if is_regression:
            folds = KFold(self.validation.cvFolds).split(X_train)
        else:
            folds = StratifiedKFold(self.validation.cvFolds).split(X_train, y_train)
        for i, (trained, validated) in enumerate(folds):
            self.recordProgress()
            self.fitAndValidate(X_train.iloc[trained], y_train.iloc[trained], X_train.iloc[validated], y_train.iloc[validated], perfClass=ModelPerformanceCV, fold=i + 1)

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

    def populateActivitySet(self, aset : models.ModelActivitySet):
        if not self.instance.predictionsType:
            raise Exception("The activity type for QSAR model predictions is not specified.")

        aset.activities.all().delete()
        molecules = aset.molecules.molecules.all()
        predictions = self.predictMolecules(molecules)

        for mol, prediction in zip(molecules, predictions):
            models.ModelActivity.objects.create(
                value=prediction,
                type=self.instance.predictionsType,
                units=self.instance.predictionsUnits,
                source=aset,
                molecule=mol,
            )

        return aset.activities.all()

