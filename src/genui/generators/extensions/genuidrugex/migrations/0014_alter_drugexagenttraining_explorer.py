# Generated by Django 4.1 on 2022-09-04 16:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('genuidrugex', '0013_drugexenvironmentscores_modifierson'),
    ]

    operations = [
        migrations.AlterField(
            model_name='drugexagenttraining',
            name='explorer',
            field=models.CharField(choices=[('GE', 'Graph Explorer (fragments)'), ('SE', 'SMILES Explorer (fragments)'), ('SM', 'SMILES Explorer (molecules)')], default='GE', max_length=2),
        ),
    ]
