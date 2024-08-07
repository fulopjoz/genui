# Generated by Django 5.0.6 on 2024-06-11 12:12

import django.db.models.deletion
import genui.utils.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('models', '0001_initial'),
        ('projects', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='model',
            name='project',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='projects.project'),
        ),
        migrations.AddField(
            model_name='model',
            name='builder',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='models.modelbuilder'),
        ),
        migrations.AddField(
            model_name='modelfile',
            name='modelInstance',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='files', to='models.model'),
        ),
        migrations.AddField(
            model_name='modelfile',
            name='format',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='models.modelfileformat'),
        ),
        migrations.AddField(
            model_name='algorithm',
            name='fileFormats',
            field=models.ManyToManyField(to='models.modelfileformat'),
        ),
        migrations.AddField(
            model_name='modelparameter',
            name='algorithm',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='parameters', to='models.algorithm'),
        ),
        migrations.CreateModel(
            name='ModelParameterBool',
            fields=[
                ('modelparametervalue_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelparametervalue')),
                ('value', models.BooleanField()),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelparametervalue',),
        ),
        migrations.CreateModel(
            name='ModelParameterFloat',
            fields=[
                ('modelparametervalue_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelparametervalue')),
                ('value', models.FloatField()),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelparametervalue',),
        ),
        migrations.CreateModel(
            name='ModelParameterInt',
            fields=[
                ('modelparametervalue_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelparametervalue')),
                ('value', models.IntegerField()),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelparametervalue',),
        ),
        migrations.CreateModel(
            name='ModelParameterStr',
            fields=[
                ('modelparametervalue_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelparametervalue')),
                ('value', models.CharField(max_length=1024)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelparametervalue',),
        ),
        migrations.AddField(
            model_name='modelparametervalue',
            name='parameter',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='models.modelparameter'),
        ),
        migrations.AddField(
            model_name='modelparametervalue',
            name='polymorphic_ctype',
            field=models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_%(app_label)s.%(class)s_set+', to='contenttypes.contenttype'),
        ),
        migrations.AddField(
            model_name='modelparameter',
            name='defaultValue',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='models.modelparametervalue'),
        ),
        migrations.CreateModel(
            name='ModelPerfomanceNN',
            fields=[
                ('modelperformance_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelperformance')),
                ('epoch', models.IntegerField()),
                ('step', models.IntegerField()),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelperformance',),
        ),
        migrations.CreateModel(
            name='ModelPerformanceCV',
            fields=[
                ('modelperformance_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelperformance')),
                ('fold', models.IntegerField()),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelperformance',),
        ),
        migrations.AddField(
            model_name='modelperformance',
            name='model',
            field=models.ForeignKey(on_delete=genui.utils.models.NON_POLYMORPHIC_CASCADE, related_name='performance', to='models.model'),
        ),
        migrations.AddField(
            model_name='modelperformance',
            name='polymorphic_ctype',
            field=models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_%(app_label)s.%(class)s_set+', to='contenttypes.contenttype'),
        ),
        migrations.AddField(
            model_name='validationstrategy',
            name='metrics',
            field=models.ManyToManyField(to='models.modelperformancemetric'),
        ),
        migrations.AddField(
            model_name='modelperformancemetric',
            name='validAlgorithms',
            field=models.ManyToManyField(related_name='metrics', to='models.algorithm'),
        ),
        migrations.AddField(
            model_name='modelperformancemetric',
            name='validModes',
            field=models.ManyToManyField(related_name='metrics', to='models.algorithmmode'),
        ),
        migrations.AddField(
            model_name='modelperformance',
            name='metric',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='models.modelperformancemetric'),
        ),
        migrations.AddField(
            model_name='trainingstrategy',
            name='algorithm',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='models.algorithm'),
        ),
        migrations.AddField(
            model_name='trainingstrategy',
            name='mode',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='models.algorithmmode'),
        ),
        migrations.AddField(
            model_name='trainingstrategy',
            name='modelInstance',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='trainingStrategies', to='models.model'),
        ),
        migrations.AddField(
            model_name='trainingstrategy',
            name='polymorphic_ctype',
            field=models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_%(app_label)s.%(class)s_set+', to='contenttypes.contenttype'),
        ),
        migrations.AddField(
            model_name='modelparametervalue',
            name='strategy',
            field=models.ForeignKey(null=True, on_delete=genui.utils.models.NON_POLYMORPHIC_CASCADE, related_name='parameters', to='models.trainingstrategy'),
        ),
        migrations.AlterUniqueTogether(
            name='modelparameter',
            unique_together={('name', 'algorithm')},
        ),
        migrations.CreateModel(
            name='ROCCurvePoint',
            fields=[
                ('modelperformance_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='models.modelperformance')),
                ('fpr', models.FloatField()),
                ('auc', models.ForeignKey(on_delete=genui.utils.models.NON_POLYMORPHIC_CASCADE, related_name='points', to='models.modelperformance')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('models.modelperformance',),
        ),
    ]
