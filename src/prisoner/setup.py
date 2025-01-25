from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'prisoner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'templates'), 
            glob('prisoner/templates/*.html')),
        (os.path.join('share', package_name, 'static'),
            glob('prisoner/static/*')),
        (os.path.join('share', package_name, 'audio'),
            glob('prisoner/audio/*.wav')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'opencv-python',
        'simpleaudio',
        'flask',
    ],
    zip_safe=True,
    maintainer='lee',
    maintainer_email='lee@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'security_cam = prisoner.security_cam:main',
            'commend_center = prisoner.commend_center:main',
            'make_db = prisoner.make_db:main',
            'AMR = prisoner.AMR:main',
            'web = prisoner.web:main',
        ],
    },
)
