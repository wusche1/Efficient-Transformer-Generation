from setuptools import setup, find_packages

setup(
    name="EfficientTransformerGeneration",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tiktoken" ,
        "pandas",
        "tqdm",
    ],
    author="Julian Schulz",
    author_email="your.email@example.com",
    #description="A short description of your package",
    #long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
    url="https://github.com/wusche1/Efficient-Transformer-Generation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
