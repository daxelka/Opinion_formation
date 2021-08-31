{ pkgs ? import <nixpkgs> {} }:
with pkgs;
let

myPythonPackage = python-packages: with python-packages; [    
    pip
    scipy
    numpy
    sklearn-deap
];

myPythonWithMyPakcage = python38.withPackages myPythonPackage;

in

pkgs.mkShell {
  name = "dev-shell";
  buildInputs = [    
    myPythonWithMyPakcage
  ];
}