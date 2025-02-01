from setuptools import setup, find_packages

setup(
    name="helmet_numberplate_detection",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "opencv-python",
        "numpy",
        "torch",
        "ultralytics",
        "pillow"
    ],
    entry_points={
        "console_scripts": [
            "helmet_numberplate_detection=hel:main"
        ]
    },
    author="Your Name",
    description="A Streamlit-based Helmet & Number Plate Detection System",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
