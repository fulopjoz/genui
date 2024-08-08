"""
tasks

Created by: Martin Sicho
On: 29-11-19, 13:44
"""

from genui.src.genui.models.models import ValidationStrategy
from genui.utils.extensions.tasks.progress import ProgressRecorder
from celery import shared_task

from .models import QSARModel, ModelActivitySet
from genui.utils.inspection import getObjectAndModuleFromFullName


@shared_task(name="BuildQSARModel", bind=True)
# CHANGE: Added 'validations' parameter to allow specifying custom validation strategies
def buildQSARModel(self, model_id, builder_class, validations=None):
    instance = QSARModel.objects.get(pk=model_id)
    builder_class = getObjectAndModuleFromFullName(builder_class)[0]
    recorder = ProgressRecorder(self)
    # CHANGE: Added logic to handle custom validation strategies
    if validations:
        # If custom validations are provided, fetch the corresponding ValidationStrategy objects
        validations = [ValidationStrategy.objects.get(pk=v_id) for v_id in validations]
    else:
        # If no custom validations are provided, use all validation strategies associated with the training strategy
        validations = list(instance.trainingStrategy.validationStrategies.all())
    builder = builder_class(
        instance,
        recorder,
        # CHANGE: Pass the validations to the builder 
        validations=validations
    )
    builder.build()

    return {
        "errors" : [repr(x) for x in builder.errors],
        "modelName" : instance.name,
        "modelID" : instance.id,
    }

@shared_task(name="PredictWithQSARModel", bind=True)
def predictWithQSARModel(self, predictions_id, builder_class):
    instance = ModelActivitySet.objects.get(pk=predictions_id)
    model = QSARModel.objects.get(pk=instance.model.id)
    builder_class = getObjectAndModuleFromFullName(builder_class)[0]
    recorder = ProgressRecorder(self)
    builder = builder_class(
        model,
        recorder
    )
    builder.populateActivitySet(instance)

    return {
        "errors" : [repr(x) for x in builder.errors],
        "modelName" : model.name,
        "modelID" : model.id,
        "activitySetName" : instance.name,
        "activitySetID" : instance.id,
    }