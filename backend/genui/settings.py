"""
Django settings for genui project.

Generated by 'django-admin startproject' using Django 2.2.7.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os
from genui import celery_app

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'euws5ei%zq!@0yyo6ta4^e3whylufayu)26th6869x=ljr44=d' if not 'GENUI_BACKEND_SECRET' in os.environ else os.environ['GENUI_BACKEND_SECRET']

# SECURITY WARNING: don't run with debug turned on in production!
if 'GENUI_BUILD_CONFIG' in os.environ and os.environ['GENUI_BUILD_CONFIG'] == 'prod':
    DEBUG = False
else:
    DEBUG = True

# determine if we are running in a docker container
DOCKER = 'DOCKER_CONTAINER' in os.environ and int(os.environ['DOCKER_CONTAINER']) == 1

if DEBUG:
    ALLOWED_HOSTS = ['*']
else:
    # TODO: modify this for production
    ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'polymorphic',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'rest_framework',
    'drf_yasg',
    'djcelery_model',
    'celery_progress',
    'django_rdkit',
    'projects.apps.ProjectsConfig',
    'compounds.apps.CompoundsConfig',
    'qsar.apps.QsarConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware', FIXME: this should be handled once we approach more serious production level
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
if DEBUG:
    MIDDLEWARE.append('genui.middleware.dev_cors_middleware')

ROOT_URLCONF = 'genui.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'genui', 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'genui.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

if DOCKER or 'GENUI_DATA_VOLUME' in os.environ:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'postgres',
            'USER': 'postgres',
            'HOST': 'db',
            'PORT': 5432,
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'postgres',
            'USER': 'postgres',
            'HOST': 'localhost',
            'PORT': 5432,
        }
    }


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = False

USE_L10N = False

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'

REACT_APP_DIR = os.path.join(BASE_DIR, '../frontend')
STATICFILES_DIRS = [
    os.path.join(REACT_APP_DIR, 'build', 'static'),
]

# rest framework
REST_FRAMEWORK = {
    # will be able to login using the normal Django Framework login views / templates
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'commons.authentication.CsrfExemptSessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ),

    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    # 'DEFAULT_PERMISSION_CLASSES': [
    #     'rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly'
    # ]
}

# celery settings
CURRENT_CELERY_APP = celery_app
if DOCKER:
    CELERY_BROKER_URL = 'redis://redis:6379'
else:
    CELERY_BROKER_URL = 'redis://localhost:6379'
# CELERY_RESULT_BACKEND = 'redis://redis:6379'
CELERY_RESULT_BACKEND = 'django-db'
CELERY_CACHE_BACKEND = 'django-cache'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_SERIALIZER = 'json'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_SEND_SENT_EVENT = True
CELERY_SEND_EVENTS = True

# FIXME:  do this somewhere
# # create the superuser if in DEBUG mode and data are supplied
# if DEBUG and 'GENUI_SUPERUSER_NAME' in os.environ:
#     from django.contrib.auth.models import User
#     User.objects.create_superuser(os.environ['GENUI_SUPERUSER_NAME'], os.environ['GENUI_SUPERUSER_EMAIL'], os.environ['GENUI_SUPERUSER_PASSWORD'])