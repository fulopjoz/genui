import json
import os

import joblib
from django.core.exceptions import ImproperlyConfigured
from rest_framework.test import APITestCase
from django.urls import reverse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from genui.compounds.models import ActivityTypes, ActivityUnits
from genui.compounds.extensions.chembl.tests import CompoundsMixIn
from genui.qsar.models import QSARModel, DescriptorGroup, ModelActivitySet
from genui.models.models import ModelPerformance, Algorithm, AlgorithmMode, ModelFile, ModelPerformanceMetric, BasicValidationStrategy
from .genuimodels import builders

            
class QSARModelInit(CompoundsMixIn):

    def setUp(self):
        super().setUp()
        self.project = self.createProject()
        self.molset = self.createMolSet(
            reverse('chemblSet-list'),
            {
                "targets": ["CHEMBL251"],
                "maxPerTarget" : 30
            }
        )

    def createTestQSARModel(
            self,
            activitySet=None,
            activityType=None,
            mode=None,
            algorithm=None,
            parameters=None,
            descriptors=None,
            metrics=None
    ):
        if not activitySet:
            activitySet = self.molset.activities.all()[0]
        if not activityType:
            activityType = ActivityTypes.objects.get(value="Ki_pChEMBL")
        if not mode:
            mode = AlgorithmMode.objects.get(name="classification")
        if not algorithm:
            algorithm = Algorithm.objects.get(name="RandomForest")
        if not parameters:
            parameters = {
                "n_estimators": 150
            }
        if not descriptors:
            descriptors = [DescriptorGroup.objects.get(name="MORGANFP")]
        if not metrics:
            metrics = [
                ModelPerformanceMetric.objects.get(name="MCC"),
                ModelPerformanceMetric.objects.get(name="ROC"),
            ]

        post_data = {
            "name": "Test Model",
            "description": "test description",
            "project": self.project.id,
            "molset": self.molset.id,
            "trainingStrategy": {
                "algorithm": algorithm.id,
                "parameters": parameters,
                "mode": mode.id,
                "descriptors": [
                    x.id for x in descriptors
                ],
                "activityThreshold": 6.5,
                "activitySet": activitySet.id,
                "activityType": activityType.id
            },
            "validationStrategies": [{
                "cvFolds": 3,
                "validSetSize": 0.2,
                "metrics": [
                    x.id for x in metrics
                ]
            }]
        }
        create_url = reverse('model-list')
        response = self.client.post(create_url, data=post_data, format='json')
        print(json.dumps(response.data, indent=4))
        self.assertEqual(response.status_code, 201)

        return QSARModel.objects.get(pk=response.data["id"])

    def predictWithModel(self, model, to_predict):
        post_data = {
            "name": f"Predictions using {model.name}",
            "molecules": to_predict.id
        }
        create_url = reverse('model-predictions', args=[model.id])
        response = self.client.post(create_url, data=post_data, format='json')
        print(json.dumps(response.data, indent=4))
        self.assertEqual(response.status_code, 201)

        instance = ModelActivitySet.objects.get(pk=response.data['id'])
        url = reverse('activitySet-activities', args=[instance.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['count'], to_predict.molecules.count())
        print(json.dumps(response.data, indent=4))

        return instance

    def uploadModel(self, filePath, algorithm, mode, descriptors, predictionsType, predictionsUnits):
        create_url = reverse('model-list')
        post_data = {
            "name": "Test Model",
            "description": "test description",
            "project": self.project.id,
            "build" : False,
            "predictionsType": predictionsType,
            "predictionsUnits": predictionsUnits,
            "trainingStrategy": {
                "algorithm": algorithm.id,
                "mode": mode.id,
                "descriptors": [
                  x.id for x in descriptors
                ]
            },
        }
        response = self.client.post(create_url, data=post_data, format='json')
        print(json.dumps(response.data, indent=4))
        self.assertEqual(response.status_code, 201)
        instance = QSARModel.objects.get(pk=response.data["id"])
        self.assertFalse(instance.modelFile)

        url = reverse('qsar-model-files-list', args=[instance.id])
        response = self.client.post(
            url,
            data={
                "file" : open(filePath, "rb"),
                "kind": ModelFile.MAIN,
            },
            format='multipart'
        )
        print(json.dumps(response.data, indent=4))
        self.assertEqual(response.status_code, 201)

        url = reverse('model-detail', args=[instance.id])
        response_other = self.client.get(url)
        self.assertEqual(response.data['file'].split('/')[-1], response_other.data['modelFile']['file'].split('/')[-1])

        return instance

class ModelInitTestCase(QSARModelInit, APITestCase):

    def test_create_view_classification(self):
        model = self.createTestQSARModel()

        path = model.modelFile.path
        model = joblib.load(model.modelFile.path)
        self.assertTrue(isinstance(model, RandomForestClassifier))

        # get the model via api
        response = self.client.get(reverse('model-list'))
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.data[0], indent=4))

        # create predictions with the model
        model = QSARModel.objects.get(pk=response.data[0]['id'])
        self.predictWithModel(model, self.molset)

        # make sure the delete cascades fine and the file gets deleted too
        self.project.delete()
        self.assertTrue(ModelPerformance.objects.count() == 0)
        self.assertTrue(not os.path.exists(path))

    def test_create_view_from_file_classification(self):
        instance_first = self.createTestQSARModel()
        self.assertEqual(instance_first.predictionsType, ActivityTypes.objects.get(value="Active Probability"))
        self.assertEqual(instance_first.predictionsUnits, None)
        instance = self.uploadModel(
            instance_first.modelFile.path,
            instance_first.trainingStrategy.algorithm,
            instance_first.trainingStrategy.mode,
            [DescriptorGroup.objects.get(name='MORGANFP')],
            instance_first.predictionsType.value,
            instance_first.predictionsUnits.value if instance_first.predictionsUnits else None
        )

        builder = builders.BasicQSARModelBuilder(instance)
        self.assertRaisesMessage(ImproperlyConfigured, "You cannot build a QSAR model with a missing validation strategy.", builder.build)
        builder.calculateDescriptors(["CC", "CCO"])
        print(builder.predict())

        activity_set = self.predictWithModel(instance, self.molset)
        for activity in activity_set.activities.all():
            self.assertEqual(activity.type, instance_first.predictionsType)
            self.assertEqual(activity.units, instance_first.predictionsUnits)

    def test_create_view_regression(self):
        model = self.createTestQSARModel(
            mode=AlgorithmMode.objects.get(name="regression"),
            metrics=ModelPerformanceMetric.objects.filter(name__in=("R2", "MSE")),
            activityType=ActivityTypes.objects.get(value="Ki")
        )
        self.assertEqual(model.predictionsType, ActivityTypes.objects.get(value="Ki"))
        self.assertEqual(model.predictionsUnits, ActivityUnits.objects.get(value="nM"))
        self.assertTrue(isinstance(joblib.load(model.modelFile.path), RandomForestRegressor))
        activity_set_orig = self.predictWithModel(model, self.molset)

        # try to upload it as a file and use that model for predictions
        model_from_file = self.uploadModel(
            model.modelFile.path,
            model.trainingStrategy.algorithm,
            model.trainingStrategy.mode,
            [DescriptorGroup.objects.get(name='MORGANFP')],
            model.predictionsType.value,
            model.predictionsUnits.value if model.predictionsUnits else None
        )
        builder = builders.BasicQSARModelBuilder(model_from_file)
        builder.calculateDescriptors(["CC", "CCO"])
        print(builder.predict())
        activity_set = self.predictWithModel(model_from_file, self.molset)
        for activity_uploaded, activity_orig in zip(activity_set.activities.all(), activity_set_orig.activities.all()):
            self.assertEqual(activity_uploaded.type, model.predictionsType)
            self.assertEqual(activity_uploaded.units, model.predictionsUnits)
            self.assertEqual(activity_uploaded.type, activity_orig.type)
            self.assertEqual(activity_uploaded.units, activity_orig.units)
            self.assertEqual(activity_uploaded.value, activity_orig.value)
               
    def test_add_validation_strategy_and_rebuild(self):
        """
        Test adding a new validation strategy to an existing QSAR model and rebuilding it.

        This test case verifies that:
        1. A new validation strategy can be added to an existing model.
        2. The model can be successfully rebuilt with multiple validation strategies.
        3. Performance metrics from both validation strategies are present after rebuilding.

        The test uses a classification model with RandomForest algorithm as an example.
        """
        # Create initial model with one validation strategy
        model = self.createTestQSARModel(
            mode=AlgorithmMode.objects.get(name="classification"),
            algorithm=Algorithm.objects.get(name="RandomForest"),
            parameters={"n_estimators": 100},
            metrics=[
                ModelPerformanceMetric.objects.get(name="ROC"),
                ModelPerformanceMetric.objects.get(name="MCC"),
            ]
        )

        # Verify that the initial model has only one validation strategy
        self.assertEqual(model.trainingStrategy.validationStrategies.count(), 1)

        # Add a second validation strategy
        second_strategy = BasicValidationStrategy.objects.create(
            cvFolds=5,
            validSetSize=0.3
        )
        # Set metrics for the second strategy
        second_strategy.metrics.set(ModelPerformanceMetric.objects.filter(name__in=["R2", "MSE"]))
        # Add the new strategy to the model's training strategy
        model.trainingStrategy.validationStrategies.add(second_strategy)

        # Rebuild the model with the new validation strategy
        response = self.client.post(reverse('model-build', args=[model.id]))
        # Check if the rebuild request was successful
        self.assertEqual(response.status_code, 200)

        # Verify that performance metrics from both validation strategies are present
        # First strategy metrics
        self.assertGreater(model.performance.filter(metric__name="ROC").count(), 0)
        self.assertGreater(model.performance.filter(metric__name="MCC").count(), 0)
        # Second strategy metrics
        self.assertGreater(model.performance.filter(metric__name="R2").count(), 0)
        self.assertGreater(model.performance.filter(metric__name="MSE").count(), 0)
        
    def test_create_model_with_multiple_strategies(self):
        """
        Test creation of a QSAR model with multiple validation strategies.

        This test case verifies that:
        1. A QSAR model can be created with multiple validation strategies via API.
        2. The created model has the correct number of validation strategies.

        The test uses a classification model with RandomForest algorithm and 
        Morgan fingerprints as descriptors.
        """
        # Prepare the data for model creation
        post_data = {
            "name": "Multi-Strategy Model",
            "description": "Model with multiple validation strategies",
            "project": self.project.id,
            "molset": self.molset.id,
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="RandomForest").id,
                "parameters": {"n_estimators": 100},
                "mode": AlgorithmMode.objects.get(name="classification").id,
                "descriptors": [DescriptorGroup.objects.get(name="MORGANFP").id],
                "activityThreshold": 6.5,
                "activitySet": self.molset.activities.all()[0].id,
                "activityType": ActivityTypes.objects.get(value="Ki_pChEMBL").id
            },
            "validationStrategies": [
                {
                    "cvFolds": 3,
                    "validSetSize": 0.2,
                    "metrics": [ModelPerformanceMetric.objects.get(name="ROC").id]
                },
                {
                    "cvFolds": 5,
                    "validSetSize": 0.3,
                    "metrics": [ModelPerformanceMetric.objects.get(name="MSE").id]
                }
            ]
        }

        # Get the URL for model creation
        create_url = reverse('model-list')

        # Send POST request to create the model
        response = self.client.post(create_url, data=post_data, format='json')

        # Check if the model was created successfully
        self.assertEqual(response.status_code, 201, "Model creation failed")

        # Retrieve the created model from the database
        model = QSARModel.objects.get(pk=response.data["id"])

        # Verify that the model has two validation strategies
        self.assertEqual(model.trainingStrategy.validationStrategies.count(), 2,
                        "Model does not have the expected number of validation strategies")
        
    def test_model_with_no_validation_strategies(self):
        """
        Test creation of a QSAR model with no validation strategies.

        This test case verifies that:
        1. Attempting to create a QSAR model without any validation strategies results in an error.
        2. The API returns a 400 Bad Request status code in this scenario.

        The test uses a classification model with RandomForest algorithm and 
        Morgan fingerprints as descriptors, but deliberately omits validation strategies.
        """
        # Prepare the data for model creation without validation strategies
        post_data = {
            "name": "No Validation Model",
            "description": "Model without validation strategies",
            "project": self.project.id,
            "molset": self.molset.id,
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="RandomForest").id,
                "parameters": {"n_estimators": 100},
                "mode": AlgorithmMode.objects.get(name="classification").id,
                "descriptors": [DescriptorGroup.objects.get(name="MORGANFP").id],
                "activityThreshold": 6.5,
                "activitySet": self.molset.activities.all()[0].id,
                "activityType": ActivityTypes.objects.get(value="Ki_pChEMBL").id
            },
            "validationStrategies": []  # Explicitly set to an empty list
        }

        # Get the URL for model creation
        create_url = reverse('model-list')

        # Attempt to create the model via POST request
        response = self.client.post(create_url, data=post_data, format='json')

        # Verify that the request is rejected with a 400 Bad Request status
        self.assertEqual(response.status_code, 400, 
                        "Expected a 400 Bad Request for model without validation strategies")

    def test_correct_performance_entries(self):
        """
        Test if the correct number of performance entries are created for a QSAR model.

        This test case verifies that:
        1. A QSAR model is created with a validation strategy.
        2. The number of performance entries matches the expected count based on
        the number of cross-validation folds and performance metrics.

        The test uses a helper method to create a test QSAR model and checks
        the resulting performance entries.
        """
        # Create a test QSAR model using a helper method
        model = self.createTestQSARModel()

        # Get the first (and possibly only) validation strategy for the model
        strategy = model.trainingStrategy.validationStrategies.first()

        # Calculate the expected number of performance entries
        # It should be the product of the number of CV folds and the number of metrics
        expected_entries = strategy.cvFolds * len(strategy.metrics.all())

        # Get the actual number of performance entries for the model
        actual_entries = model.performance.count()

        # Assert that the actual number of entries matches the expected number
        self.assertEqual(actual_entries, expected_entries,
                        f"Expected {expected_entries} performance entries, but found {actual_entries}")

    def test_different_metric_combinations(self):
        """
        Test creation and rebuilding of a QSAR model with multiple validation strategies and different metric combinations.

        This test case verifies that:
        1. A QSAR model can be created with multiple validation strategies, each with different metrics.
        2. The model can be successfully rebuilt.
        3. After rebuilding, performance entries for all specified metrics are present.

        The test uses a classification model with RandomForest algorithm and Morgan fingerprints as descriptors.
        It includes two validation strategies with different metrics for each.
        """
        # Prepare the data for model creation
        post_data = {
            "name": "Multi-Metric Model",
            "description": "Model with different metric combinations",
            "project": self.project.id,
            "molset": self.molset.id,
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="RandomForest").id,
                "parameters": {"n_estimators": 100},
                "mode": AlgorithmMode.objects.get(name="classification").id,
                "descriptors": [DescriptorGroup.objects.get(name="MORGANFP").id],
                "activityThreshold": 6.5,
                "activitySet": self.molset.activities.all()[0].id,
                "activityType": ActivityTypes.objects.get(value="Ki_pChEMBL").id
            },
            "validationStrategies": [
                {
                    "cvFolds": 3,
                    "validSetSize": 0.2,
                    "metrics": [
                        ModelPerformanceMetric.objects.get(name="ROC").id,
                        ModelPerformanceMetric.objects.get(name="MCC").id
                    ]
                },
                {
                    "cvFolds": 5,
                    "validSetSize": 0.3,
                    "metrics": [
                        ModelPerformanceMetric.objects.get(name="R2").id,
                        ModelPerformanceMetric.objects.get(name="MSE").id
                    ]
                }
            ]
        }

        # Get the URL for model creation
        create_url = reverse('model-list')

        # Send POST request to create the model
        response = self.client.post(create_url, data=post_data, format='json')

        # Check if the model was created successfully
        self.assertEqual(response.status_code, 201, "Model creation failed")

        # Retrieve the created model from the database
        model = QSARModel.objects.get(pk=response.data["id"])

        # Get the URL for rebuilding the model
        rebuild_url = reverse('model-build', args=[model.id])

        # Send POST request to rebuild the model
        rebuild_response = self.client.post(rebuild_url)

        # Check if the model was rebuilt successfully
        self.assertEqual(rebuild_response.status_code, 200, "Model rebuild failed")

        # Check if performance entries for all metrics were calculated
        metrics_to_check = ["ROC", "MCC", "R2", "MSE"]
        for metric_name in metrics_to_check:
            performance_count = model.performance.filter(metric__name=metric_name).count()
            self.assertGreater(performance_count, 0, 
                            f"No performance entries found for metric: {metric_name}")