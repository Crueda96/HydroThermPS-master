from setuptools import setup

setup(name='HydroThermPS', version='1.0.0', packages=['HydroThermPS'], 
entry_points= {
    'console_scripts': ['HydroThermPS = HydroThermPS.__main__:main']
})
