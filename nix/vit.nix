{
  lib,
  # build deps
  buildPythonPackage,
  fetchFromGitHub,
  # Py build
  setuptools,
  # Deps
  torch,
  timm,
  transformers,
  jupyter,
}:
buildPythonPackage {
  pname = "vit";
  version = "0.0.1";

  src = ./..; # For local testing, add flag --impure when running
  # src = fetchFromGitHub {
  #   owner = "afermg";
  #   repo = "baby";
  #   rev = "39eec0d4c3b8fad9b0a8683cbedf9b4558e07222";
  #   sha256 = "sha256-ptLXindgixDa4AV3x+sQ9I4W0PScIQMkyMNMo0WFa0M=";
  # };

  pyproject = true;
  buildInputs = [
    # setuptools-scm
    setuptools
  ];
  propagatedBuildInputs = [
    torch
    timm
    transformers
    jupyter
  ];

  pythonImportsCheck = [
  ];

  meta = {
    description = "Generic environment for visual transformers.";
    homepage = "https://github.com/afermg/nahual_vit";
    license = lib.licenses.mit;
  };
}
