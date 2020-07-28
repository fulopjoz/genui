# Generated by Django 2.2.8 on 2020-07-13 14:42

from django.db import migrations, models
import django.db.models.deletion
import genui.utils.models


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('compounds', '0003_auto_20200512_0817'),
    ]

    operations = [
        migrations.CreateModel(
            name='MolSetFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(storage=genui.utils.models.OverwriteStorage(), upload_to='compounds/sets/files/')),
                ('molset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='files', to='compounds.MolSet')),
                ('polymorphic_ctype', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_compounds.molsetfile_set+', to='contenttypes.ContentType')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
    ]