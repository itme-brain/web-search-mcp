{
  description = "Fully-FOSS web-search MCP server: SearXNG + Crawl4AI + FlashRank bundled as one docker-compose stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        libPath = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.file];
        python = pkgs.python312.override {
          packageOverrides = self: super: {
            fastmcp = super.fastmcp.overridePythonAttrs (_: {
              doCheck = false;
            });
            python-docx = super.python-docx.overridePythonAttrs (_: {
              doCheck = false;
            });
            flashrank = self.buildPythonPackage rec {
              pname = "flashrank";
              version = "0.2.10";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/55/1f/176cb4a857a70c3538f637e19389ab6aed21548a1ba1d1424fccc8bba108/FlashRank-0.2.10.tar.gz";
                sha256 = "f8f82a25c32fdfc668a09dc4089421d6aab8e7f71308424b541f40bb3f01d9db";
              };
              pyproject = true;
              build-system = [ self.setuptools ];
              propagatedBuildInputs = with self; [
                numpy
                onnxruntime
                requests
                tokenizers
                tqdm
              ];
              doCheck = false;
            };
          };
        };

        # Tools needed at runtime by the deploy script. Docker daemon must be
        # supplied by the host (Nix does not install a daemon on non-NixOS).
        runtimeTools = [
          python
          pkgs.docker-client
          pkgs.docker-compose
          pkgs.just
          pkgs.curl
          pkgs.jq
          pkgs.coreutils
          pkgs.openssl
          pkgs.gnused
          pkgs.uv
          pkgs.file
        ];

        deploy = pkgs.writeShellApplication {
          name = "web-search-mcp-deploy";
          runtimeInputs = runtimeTools;
          text = ''
            set -euo pipefail

            if [[ ! -f docker-compose.yml ]]; then
              echo "error: run this from the web-search-mcp repo root (no docker-compose.yml in $PWD)" >&2
              exit 1
            fi

            if ! docker info >/dev/null 2>&1; then
              echo "error: cannot talk to the Docker daemon." >&2
              echo "  On non-NixOS, install Docker on the host (https://docs.docker.com/engine/install/)." >&2
              echo "  Ensure your user is in the 'docker' group, or re-run with sudo." >&2
              exit 1
            fi

            if [[ ! -f .env ]]; then
              echo "info: no .env found — copying env.sample" >&2
              cp env.sample .env
            fi

            if [[ ! -f searxng/config/settings.yml ]]; then
              echo "info: rendering searxng/config/settings.yml with a random secret_key" >&2
              sed "s|ultrasecretkey|$(openssl rand -hex 32)|" \
                searxng/config/settings.yml.template \
                > searxng/config/settings.yml
            fi

            # Regenerate the pip lock if the source is newer. Keeps the
            # committed lockfile aligned with requirements.in without
            # silently re-resolving transitives on every deploy.
            if [[ requirements.in -nt requirements.txt ]]; then
              echo ">> requirements.in is newer than requirements.txt — regenerating lock via uv"
              uv pip compile --quiet --prerelease=allow --generate-hashes requirements.in -o requirements.txt
            fi

            echo ">> building + starting stack"
            docker compose up -d --build

            echo ">> waiting for MCP to become ready"
            # shellcheck disable=SC1091
            source .env
            port="''${MCP_HOST_PORT:-8002}"
            for _ in {1..60}; do
              if curl -sf "http://localhost:$port/ready" >/dev/null 2>&1; then
                echo ">> MCP ready on http://localhost:$port"
                exit 0
              fi
              sleep 1
            done
            echo "warn: MCP did not respond within 60s. Check 'just logs'." >&2
            exit 1
          '';
        };

        teardown = pkgs.writeShellApplication {
          name = "web-search-mcp-teardown";
          runtimeInputs = runtimeTools;
          text = ''
            set -euo pipefail
            if [[ ! -f docker-compose.yml ]]; then
              echo "error: run this from the web-search-mcp repo root" >&2
              exit 1
            fi
            docker compose down "$@"
          '';
        };
      in
      {
        packages = {
          inherit deploy teardown;
          default = deploy;
        };

        apps = {
          deploy = flake-utils.lib.mkApp { drv = deploy; };
          teardown = flake-utils.lib.mkApp { drv = teardown; };
          default = flake-utils.lib.mkApp { drv = deploy; };
        };

        devShells.default = pkgs.mkShell {
          name = "web-search-mcp-dev";
          buildInputs = runtimeTools;
          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export UV_PROJECT_ENVIRONMENT=.venv
            if [[ -x .venv/bin/python ]]; then
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
            fi
          '';
        };
      });
}
