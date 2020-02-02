"""
tasks

Created by: Martin Sicho
On: 28-01-20, 13:52
"""
from celery import shared_task

from commons.tasks import ProgressRecorder
from generators import models

from generators.core import builders


@shared_task(name="buildDrugExNet", bind=True)
def buildDrugExNet(self, model_id, builder_class):
    instance = models.DrugExNet.objects.get(pk=model_id)
    builder_class = getattr(builders, builder_class)
    recorder = ProgressRecorder(self)
    builder = builder_class(
        instance,
        progress=recorder
    )
    builder.build()

    return {
        "errors" : [repr(x) for x in builder.errors],
        "modelFile" : instance.modelFile.path,
        "corpusFile" : instance.corpus.corpusFile.path,
        "vocFile" : instance.corpus.vocFile.path
    }

@shared_task(name="buildDrugExAgent", bind=True)
def buildDrugExAgent(self, model_id, builder_class):
    instance = models.DrugExAgent.objects.get(pk=model_id)
    builder_class = getattr(builders, builder_class)
    recorder = ProgressRecorder(self)
    builder = builder_class(
        instance,
        progress=recorder
    )
    builder.build()

    return {
        "errors" : [repr(x) for x in builder.errors],
        "modelFile" : instance.modelFile.path,
    }
