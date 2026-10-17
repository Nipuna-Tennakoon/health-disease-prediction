"""Reusable pipeline code for the health disease prediction project.

Modules map one-to-one to pipeline stages so they can run from the CLI or as Airflow tasks:
preprocessing -> features/training -> inference.
"""
