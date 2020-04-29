# Generated by Django 2.2.8 on 2020-04-27 20:39

from django.db import migrations, models
import genui.commons.models


class Migration(migrations.Migration):

    dependencies = [
        ('qsar', '0001_initial'),
        ('generators', '0001_initial'),
        ('maps', '0001_initial'),
        ('compounds', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chemblactivity',
            name='activity_ptr',
        ),
        migrations.RemoveField(
            model_name='chemblactivity',
            name='assay',
        ),
        migrations.RemoveField(
            model_name='chemblactivity',
            name='target',
        ),
        migrations.RemoveField(
            model_name='chemblcompounds',
            name='molset_ptr',
        ),
        migrations.RemoveField(
            model_name='chemblcompounds',
            name='targets',
        ),
        migrations.RemoveField(
            model_name='chemblmolecule',
            name='molecule_ptr',
        ),
        migrations.AlterField(
            model_name='moleculepic',
            name='image',
            field=models.ImageField(storage=genui.commons.models.OverwriteStorage(), upload_to='compounds/pics/'),
        ),
        migrations.DeleteModel(
            name='ChEMBLActivities',
        ),
        migrations.DeleteModel(
            name='ChEMBLActivity',
        ),
        migrations.DeleteModel(
            name='ChEMBLAssay',
        ),
        migrations.DeleteModel(
            name='ChEMBLCompounds',
        ),
        migrations.DeleteModel(
            name='ChEMBLMolecule',
        ),
        migrations.DeleteModel(
            name='ChEMBLTarget',
        ),
    ]
