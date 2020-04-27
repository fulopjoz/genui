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
import genui.apps

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join('../', __file__))))

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

try:
    HOST_ROOT = f"{os.environ['GENUI_BACKEND_PROTOCOL']}://{os.environ['GENUI_BACKEND_HOST']}:{os.environ['GENUI_BACKEND_PORT']}"
except KeyError:
    HOST_ROOT = ''

if DEBUG:
    ALLOWED_HOSTS = ['*']
else:
    # TODO: modify this for production
    ALLOWED_HOSTS = []

SITE_ID = 1

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'polymorphic',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'corsheaders',
    'rest_framework',
    'rest_framework.authtoken',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'rest_auth',
    'rest_auth.registration',
    'drf_yasg',
    'django_celery_results',
    'djcelery_model',
    'celery_progress',
    'django_rdkit',
] + genui.apps.ALL

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware', FIXME: this should be handled once we approach more serious production level
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
if not DOCKER and DEBUG:
    CORS_ORIGIN_WHITELIST = [
    "http://localhost:3000",
    ]
    CORS_ALLOW_CREDENTIALS = True
    SESSION_COOKIE_SAMESITE = None

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

if DOCKER:
    # FIXME: set user and password from environment variables
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
            'NAME': 'genui',
            'USER': 'genui',
            'PASSWORD': 'genui',
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

# E-mail settings
if DEBUG:
    EMAIL_HOST = 'localhost'
    EMAIL_PORT = 1025
    EMAIL_HOST_USER = ''
    EMAIL_HOST_PASSWORD = ''
    EMAIL_USE_TLS = False
    DEFAULT_FROM_EMAIL = 'testing@example.com'

# Accounts
ACCOUNT_EMAIL_VERIFICATION = 'none' # TODO: if we decide to expose this to the public somehow, we should make this 'mandatory'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_CONFIRM_EMAIL_ON_GET = True
LOGIN_REDIRECT_URL = '/' # TODO: should be changed if frontend is hosted on a different host

# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = False

USE_L10N = False

USE_TZ = True

# Media files
MEDIA_URL = f'{HOST_ROOT}/downloads/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static/')

# REACT_APP_DIR = os.path.join(BASE_DIR, '../frontend')
# STATICFILES_DIRS = [
#     os.path.join(REACT_APP_DIR, 'build', 'static'),
# ]

# rest framework
REST_FRAMEWORK = {
    # will be able to login using the normal Django Framework login views / templates
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'genui.commons.authentication.CsrfExemptSessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ),
    'URLS_ROOT' : 'auth/',

    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.DjangoModelPermissions'
    ]
}
SWAGGER_SETTINGS = {
    'SECURITY_DEFINITIONS': {
        'api_key': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization'
        }
    },
    'LOGIN_URL' : f'{REST_FRAMEWORK["URLS_ROOT"]}login/',
    'LOGOUT_URL' : f'{REST_FRAMEWORK["URLS_ROOT"]}logout/',
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

# genui specific settings
GENUI_MODEL_APPS = [
    "genui.generators",
    "genui.qsar",
    "genui.maps"
]