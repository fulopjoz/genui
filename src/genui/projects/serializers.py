"""
serializers

Created by: Martin Sicho
On: 05-12-19, 12:25
"""
from django.conf import settings

from genui import apps

from .models import Project
from rest_framework import serializers

# Serializers define the API representation.
from .tasks import createDefaultModels
from ..utils.extensions.tasks.utils import runTask


class ProjectSerializer(serializers.HyperlinkedModelSerializer):
    owner = serializers.HiddenField(
        default=serializers.CurrentUserDefault()
    )

    class Meta:
        model = Project
        fields = ('id', 'name', 'description', 'created', 'updated', 'owner')
        read_only_fields = ('created', 'updated', 'owner')

    def create(self, validated_data):
        ret = super().create(validated_data)

        for app in apps.all_():
            runTask(
                    createDefaultModels,
                    eager=hasattr(settings, 'CELERY_TASK_ALWAYS_EAGER') and settings.CELERY_TASK_ALWAYS_EAGER,
                    args=(
                        ret.id,
                        app
                    ),
                )

        return ret


class FilterToProjectMixIn:

    def get_queryset(self):
        queryset = super().get_queryset()
        project = self.request.query_params.get('project_id', None)
        if project is not None:
            queryset = queryset.filter(project__pk=int(project))
        return queryset