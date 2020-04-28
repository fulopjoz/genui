from genui.commons.views import FilterToUserMixIn
from genui.maps.core.builders import MapBuilder
from genui.modelling.core.bases import Algorithm
from genui.modelling.views import ModelViewSet, AlgorithmViewSet, FilterToModelMixin
from . import models, serializers, tasks
from rest_framework import generics, pagination

class MapViewSet(ModelViewSet):
    queryset = models.Map.objects.all()
    serializer_class = serializers.MapSerializer
    init_serializer_class = serializers.MapInitSerializer
    builder_class = MapBuilder
    build_task = tasks.createMap

class MappingAlgViewSet(AlgorithmViewSet):

    def get_queryset(self):
        current = super().get_queryset()
        return current.filter(validModes__name__in=(Algorithm.MAP,)).distinct('id')

class PointPagination(pagination.PageNumberPagination):
    page_size = 30

class PointsView(
    FilterToModelMixin,
    FilterToUserMixIn,
    generics.ListAPIView
):
    queryset = models.Point.objects.order_by('id')
    serializer_class = serializers.PointSerializer
    pagination_class = PointPagination
    lookup_field = "map"
    owner_relation = "map__project__owner"