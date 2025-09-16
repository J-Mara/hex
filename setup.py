from setuptools import setup, Extension

ext = Extension(
    name="hexfast",
    sources=["hexfast.c"],
    extra_compile_args=["-O3", "-std=c11"],  # you can add "-march=native" on Intel if you like
)

setup(
    name="hexfast",
    version="0.1.0",
    description="C accelerators for Hex (evaluator + simple players)",
    ext_modules=[ext],
)
