from setuptools import setup, Extension

ext = Extension(
    name="symnmfc",                      
    sources=["symnmfmodule.c", "symnmf.c"],   
    extra_compile_args=["-O3", "-std=c99", "-Wall", "-Wextra"],
    libraries=["m"],                         
)

setup(
    name="symnmf_pkg",
    version="0.1.0",
    ext_modules=[ext],
)
