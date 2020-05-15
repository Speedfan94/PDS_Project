from setuptools import setup

setup(
    name='PDS_Project',
    version='0.0.1dev1',
    description="Semester Project - Programming Data Science",
    author="team-we-hend-stroh-to-be-the-best-ulk-nudel",
    author_email="student@uni-koeln.de",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click', 'shapely', 'folium'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:main']
    }
)
