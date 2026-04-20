{
  description = "Fully-FOSS web-search MCP server: SearXNG + Crawl4AI + Jina reranker bundled as one docker-compose stack";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
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
              version = "0.2.9";
              src = pkgs.fetchurl {
                url = "https://files.pythonhosted.org/packages/0c/8c/4b44180d4be0f93bffe31db7229c727638994c74f04257f3844bca066b88/FlashRank-0.2.9.tar.gz";
                sha256 = "475f1192e0722da1a4409812165ebc7e3eccec56e7b7853ed9dd5dd5c9c985f5";
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
        pythonEnv = python.withPackages (ps: with ps; [
          fastmcp
          flashrank
          httpx
          python-docx
          pypdf
          trafilatura
          pytest
          pytest-asyncio
        ]);

        # Tools needed at runtime by the deploy script. Docker daemon must be
        # supplied by the host (Nix does not install a daemon on non-NixOS).
        runtimeTools = [
          pkgs.docker-client
          pkgs.docker-compose
          pkgs.just
          pkgs.curl
          pkgs.jq
          pkgs.coreutils
          pkgs.openssl
          pkgs.gnused
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
          buildInputs = runtimeTools ++ [pythonEnv];
          shellHook = ''
            echo "web-search-mcp devshell — try: just, just up, just test"
          '';
        };
      });
}
