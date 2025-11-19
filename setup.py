"""
Setup script for FlashAttention implementation.
"""

from setuptools import find_packages, setup

setup(
    name="flashattention-edu",
    version="1.0.0",
    description="Educational implementation of FlashAttention algorithm",
    author="Shuang Ma, Pablo Rodriguez, Pei Yu Lin",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "matplotlib>=3.5.0",
    ],
)
