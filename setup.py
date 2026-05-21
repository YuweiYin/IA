from setuptools import find_packages, setup


def get_requirements(path: str):
    return [_l.strip() for _l in open(path)]


setup(
    name="ia",
    version="1.0",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
