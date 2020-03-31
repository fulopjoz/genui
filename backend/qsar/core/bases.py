"""
core

Created by: Martin Sicho
On: 14-01-20, 10:16
"""

import modelling.models
from commons.helpers import findClassInModule
from compounds.models import Molecule, ActivityTypes
from modelling.core.bases import Algorithm, CompleteBuilder
from qsar import models
import pandas as pd
from pandas import DataFrame, Series


class DescriptorCalculator:
    group_name = None

    def __init__(self, builder):
        self.builder = builder

    def __call__(self, smiles):
        pass

    @classmethod
    def getDjangoModel(cls) -> models.DescriptorGroup:
        if not cls.group_name:
            raise Exception('You have to specify a name for the descriptor group in its class "group_name" property')
        return models.DescriptorGroup.objects.get_or_create(name=cls.group_name)[0]

class DescriptorBuilderMixIn:

    @staticmethod
    def findDescriptorClass(name):
        from . import descriptors
        return findClassInModule(DescriptorCalculator, descriptors, "group_name", name)

    def __init__(self, instance: models.Model, progress=None, onFitCall=None):
        super().__init__(instance, progress, onFitCall)
        self.molsets = [self.instance.molset] if hasattr(self.instance, "molset") else self.instance.molsets.all()
        self.descriptorClasses = [self.findDescriptorClass(x.name) for x in self.training.descriptors.all()]

        self.X = None
        self.y = None

    def calculateDescriptors(self, mols=None):
        """
        Calculate descriptors for the given molecules
        and save them as X in this instance. If mols is None,
        the 'self.mols' or 'self.molsets' (in the order of pereference)
        attributes will be used to get molecules for the calculation.

        :param mols: List of molecules to save as X. Can be either instances of Molecule or smiles strings
        :return:
        """

        if mols is not None:
            smiles = [x.canonicalSMILES if isinstance(x, Molecule) else x for x in mols]
        elif hasattr(self, "mols"):
            smiles = [x.canonicalSMILES if isinstance(x, Molecule) else x for x in self.mols]
        elif hasattr(self, "molsets"):
            smiles = []
            for molset in self.molsets:
                for mol in molset.molecules.all():
                    smiles.append(mol.canonicalSMILES)
        else:
            raise Exception("No molecules to calculate descriptors from.")

        self.X = DataFrame()
        for desc_class in self.descriptorClasses:
            calculator = desc_class(self)
            temp = calculator(smiles)
            temp.columns = [f"{desc_class.group_name}_{x}" for x in temp.columns]
            self.X = pd.concat([self.X, temp], axis=1)
        return self.X

class QSARModelBuilder(DescriptorBuilderMixIn, CompleteBuilder):

    def getX(self) -> DataFrame:
        return self.X

    def getY(self) -> Series:
        return self.y

    def saveActivities(self):
        if not self.getY():
            molset = self.molsets[0] # FIXME: allow processing of multiple sets
            activity_sets = [x for x in molset.activities.all() if not isinstance(x, models.ModelActivitySet)]

            if not activity_sets:
                raise Exception("Could not find any activity data.")

            compounds = []
            activities = []
            for aset in activity_sets:
                cmpds, acs = aset.cleanForModelling() # TODO: we should specify the activity type and units during this call
                compounds.extend(cmpds)
                activities.extend(acs)

            if self.training.mode.name == Algorithm.CLASSIFICATION:
                activity_thrs = self.training.activityThreshold
                activities = Series(activities).apply(lambda x : 1 if x >= activity_thrs else 0)
            self.y = activities

            if self.training.mode.name == modelling.core.bases.Algorithm.REGRESSION:
                # TODO: this should already be set when the model is created and we should feed it to cleanForModelling above to get the right data or an exception if something doesn't check out
                self.training.modelledActivityType = molset.modelledActivityType
                self.training.modelledActivityUnits = molset.modelledActivityUnits
            else:
                self.training.modelledActivityType = ActivityTypes.objects.get_or_create(
                    value="ActivePrb"
                )[0]
                self.training.modelledActivityUnits = None
            self.training.save()

            return self.y, compounds

    def fitAndValidate(
            self,
            X_train : DataFrame,
            y_train : Series,
            X_validated : DataFrame,
            y_validated : Series,
            y_predicted=None,
            perfClass=modelling.models.ModelPerformance,
            *args,
            **kwargs
    ):
        if self.training.mode.name == Algorithm.CLASSIFICATION:
            model = self.algorithmClass(self)
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_validated)
            y_predicted = [1 if x >= 0.5 else 0 for x in y_predicted]
        super().fitAndValidate(X_train, y_train, X_validated, y_validated, y_predicted, perfClass, *args, **kwargs)

    def populateActivitySet(self, aset : models.ModelActivitySet):
        raise NotImplementedError(f"Every QSAR model builder has to implement the {self.populateActivitySet.__name__} method.")
