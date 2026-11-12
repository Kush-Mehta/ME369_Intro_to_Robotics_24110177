import os
from glob import glob

from setuptools import setup

package_name = 'omx_calibration'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Parshva Patel',
    maintainer_email='parshvapatel1456@gmail.com',
    description='Guided calibration of OpenMANIPULATOR-X + RealSense peg vision',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wizard = omx_calibration.wizard:main',
            'preflight = omx_calibration.preflight:main',
            'go_home = omx_calibration.go_home:main',
            'pick_place = omx_calibration.pick_place:main',
            'fit_camera = omx_calibration.fit_camera:main',
        ],
    },
)
