from setuptools import setup, find_packages

setup(
    name='KosmosX',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='MIT',
    description='Kosmos-X - PyTorch',
    author='Kye Gomez',
    author_email='kye@apac.ai',
    long_description_content_type='text/markdown',
    url='https://github.com/kyegomez/Kosmos-X',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'optimizers',
        'Prompt Engineering'
    ],
    install_requires=[
        'transformers',
        'torch',
        'bitsandbytes',
        'flamingo_pytorch',
        'pillow',
        'accelerate',
        'datasets',
        'rich',
        'tensorboard',
        'wandb',
        'boto3',
        'lion-pytorch',
        'fairscale'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
