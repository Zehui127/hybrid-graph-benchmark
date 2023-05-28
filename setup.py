from setuptools import setup, find_packages

setup(
    name="hybrid-graph",
    version="0.2",
    packages=find_packages(),
    package_data={'hg': ['datasets/dataset_info.yaml']},
    install_requires=[
        'torch',
        'torch_scatter',
        'torch_sparse',
        'torch_geometric==2.2.0',
        'ipdb==0.13.13',
        'ipython==8.13.2',
        'matplotlib==3.7.1',
        'networkx==3.1',
        'numpy==1.24.2',
        'pytorch_lightning==1.9.3',
        'PyYAML==6.0',
        'scikit_learn==1.2.1',
        'seaborn==0.12.2',
        'torchmetrics==0.11.1'],
    entry_points={
        'console_scripts': [
            'hybrid-graph = hg.hybrid_graph.cli:main',
        ],
    },
    author="Xiangyu Zhao;Zehui Li",
    author_email="zehui.li22@imperial.ac.uk;x.zhao22@imperial.ac.uk",
    description="A python package for accessing and Hybrid-graph Datasets and train-eval framework for GNNs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Zehui127/hypergraph-benchmarks/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
