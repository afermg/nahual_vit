{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        modelPackages = {
          vit = pkgs.python3.pkgs.callPackage ./nix/vit.nix {};
        };
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
          modelPackages.vit
        ]);
        runMorphem = pkgs.writeScriptBin "nahual-morphem" ''
          #!${pkgs.bash}/bin/bash
          exec ${python_with_pkgs}/bin/python ${self}/src/vit/morphem.py "''${1:-tcp://0.0.0.0:5555}"
        '';
        runOpenphenom = pkgs.writeScriptBin "nahual-openphenom" ''
          #!${pkgs.bash}/bin/bash
          exec ${python_with_pkgs}/bin/python ${self}/src/vit/openphenom.py "''${1:-tcp://0.0.0.0:5555}"
        '';
        runContainer = pkgs.writeScriptBin "nahual-vit" ''
          #!${pkgs.bash}/bin/bash
          variant="''${1:-morphem}"
          if [[ $# -gt 0 ]]; then shift; fi
          address="''${1:-tcp://0.0.0.0:5555}"
          case "$variant" in
            morphem) exec ${runMorphem}/bin/nahual-morphem "$address" ;;
            openphenom) exec ${runOpenphenom}/bin/nahual-openphenom "$address" ;;
            *) echo "unknown ViT variant: $variant (expected morphem or openphenom)" >&2; exit 2 ;;
          esac
        '';
        appsForSystem = rec {
          morphem = {
            type = "app";
            program = "${runMorphem}/bin/nahual-morphem";
          };
          openphenom = {
            type = "app";
            program = "${runOpenphenom}/bin/nahual-openphenom";
          };
          default = morphem;
        };
      in
        with pkgs; rec {
          scripts = {
            inherit runMorphem runOpenphenom;
          };
          apps = appsForSystem;
          packages =
            modelPackages
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "vit";
                title = "Nahual ViT";
                description = "MorphEM and OpenPhenom feature extraction served through Nahual";
                source = "https://github.com/afermg/nahual_vit";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runContainer;
                entrypoint = "${runContainer}/bin/nahual-vit";
                cmd = [
                  "morphem"
                  "tcp://0.0.0.0:5555"
                ];
              };
            };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              python3Packages.venvShellHook
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
            ];
            currentSystem = system;
            venvDir = "./.venv";
            postVenvCreation = ''unset SOURCE_DATE_EPOCH'';
            postShellHook = ''unset SOURCE_DATE_EPOCH'';
            shellHook = ''
              runHook venvShellHook
              export PYTHONSAFEPATH=1
              export PYTHONDONTWRITEBYTECODE=1
            '';
          };
        }
    );
}
