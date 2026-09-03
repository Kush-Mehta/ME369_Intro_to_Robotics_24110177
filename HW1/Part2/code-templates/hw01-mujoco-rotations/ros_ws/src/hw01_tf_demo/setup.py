from setuptools import find_packages, setup

package_name = "hw01_tf_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Madhu Vadali",
    maintainer_email="madhu.vadali@iitgn.ac.in",
    description=(
        "ME 639 HW1 Task 3: TF broadcast demo for current-frame vs. "
        "fixed-frame rotation composition."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tf_broadcaster_node = hw01_tf_demo.tf_broadcaster_node:main",
        ],
    },
)
