# NnUNetDashboard
#
# Scripted module for 3D Slicer. Can also be loaded without building Slicer
# via Edit → Application Settings → Modules → Additional module paths.

cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(NnUNetDashboard)

#-----------------------------------------------------------------------------
set(MODULE_NAME NnUNetDashboard)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  NnUNetDashboardLib/__init__.py
  NnUNetDashboardLib/NnUNetClient.py
  NnUNetDashboardLib/ModuleSettings.py
  NnUNetDashboardLib/PredictionTool.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/requirements.txt
  Resources/Icons/NnUNetDashboard.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
