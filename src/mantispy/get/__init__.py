"""Accessors that turn a mantispy AnnData into plain Python objects."""

from scanpy.get import obs_df, var_df

from mantispy.get._accessors import controls, features, to_dataframe

__all__ = ["controls", "features", "obs_df", "to_dataframe", "var_df"]
