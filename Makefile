# Variables
REPO_URL      := github.com/habedi/vq
BINARY_NAME   := $(or $(PROJ_BINARY), $(notdir $(REPO_URL)-examples))
BINARY        := $(BINARY_NAME)
MAKEFILE_LIST := Makefile
CARGO_CMD     := cargo
CARGO_TERM_COLOR := always
RUST_BACKTRACE := 0
RUST_LOG      := info
DEBUG_VQ      := 0

# Pass debug variables to cargo commands
DEBUG_ARGS    := DEBUG_VQ=$(DEBUG_VQ) RUST_BACKTRACE=$(RUST_BACKTRACE)
PYVQ_DIR      := pyvq
WHEEL_DIR     := dist

# Adjust PATH if necessary (append /snap/bin if not present)
PATH          := $(if $(findstring /snap/bin,$(PATH)),$(PATH),/snap/bin:$(PATH))

# Find the latest built Python wheel file
WHEEL_FILE    := $(shell ls $(PYVQ_DIR)/$(WHEEL_DIR)/pyvq-*.whl 2>/dev/null | head -n 1)

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show the help message for each target
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

########################################################################################
## Rust Targets
########################################################################################

.PHONY: format
format: ## Format Rust files
	@echo "Formatting Rust files..."
	@$(CARGO_CMD) fmt

.PHONY: test
test: format ## Run the tests
	@echo "Running tests..."
	@$(DEBUG_ARGS) $(CARGO_CMD) test -- --nocapture

.PHONY: coverage
coverage: format ## Generate test coverage report
	@echo "Generating test coverage report..."
	@$(DEBUG_ARGS) $(CARGO_CMD) tarpaulin --out Xml --out Html

.PHONY: build
build: format ## Build the binary for the current platform
	@echo "Building the project..."
	@$(DEBUG_ARGS) $(CARGO_CMD) build --release --features binaries

.PHONY: run
run: build ## Build and run the binary
	@echo "Running the $(BINARY) binary..."
	@$(DEBUG_ARGS) $(CARGO_CMD) run --release --features binaries --bin $(BINARY)

.PHONY: clean
clean: ## Remove generated and temporary files
	@echo "Cleaning up..."
	@$(CARGO_CMD) clean
	@cd $(PYVQ_DIR) && $(CARGO_CMD) clean
	@rm -f benchmark_results.csv eval_*.csv

.PHONY: install-snap
install-snap: ## Install a few dependencies using Snapcraft
	@echo "Installing the snap package..."
	@sudo apt-get update
	@sudo apt-get install -y snapd
	@sudo snap refresh
	@sudo snap install rustup --classic

.PHONY: install-deps
install-deps: install-snap ## Install development dependencies
	@echo "Installing dependencies..."
	@rustup component add rustfmt clippy
	@$(CARGO_CMD) install cargo-tarpaulin
	@$(CARGO_CMD) install cargo-audit
	@$(CARGO_CMD) install cargo-nextest

.PHONY: lint
lint: format ## Run the linters
	@echo "Linting Rust files..."
	@$(DEBUG_ARGS) $(CARGO_CMD) clippy -- -D warnings

.PHONY: publish
publish: ## Publish the package to crates.io (needs CARGO_REGISTRY_TOKEN to be set)
	@echo "Publishing the package to Cargo registry..."
	@$(CARGO_CMD) publish --token $(CARGO_REGISTRY_TOKEN)

.PHONY: bench
bench: ## Run the benchmarks
	@echo "Running benchmarks..."
	@$(DEBUG_ARGS) $(CARGO_CMD) bench

.PHONY: eval
eval: ## Evaluate an implementation (ALG must be: bq, sq, pq, or tsvq)
	@echo && \
	if [ -z "$(ALG)" ]; then \
	  echo "Please provide the ALG argument"; exit 1; \
	fi
	@echo "Evaluating implementation with argument: $(ALG)"
	@$(CARGO_CMD) run --release --features binaries --bin eval -- --eval $(ALG)

.PHONY: eval-all
eval-all: ## Evaluate all the implementations (bq, sq, pq, and tsvq)
	@echo "Evaluating all implementations..."
	@$(MAKE) eval ALG=bq
	@$(MAKE) eval ALG=sq
	@$(MAKE) eval ALG=pq
	@$(MAKE) eval ALG=tsvq

.PHONY: fix-lint
fix-lint: ## Fix the linter warnings
	@echo "Fixing linter warnings..."
	@$(CARGO_CMD) clippy --fix --allow-dirty --allow-staged --all-targets --workspace --all-features -- -D warnings

.PHONY: nextest
nextest: ## Run tests using nextest
	@echo "Running tests using nextest..."
	@$(DEBUG_ARGS) $(CARGO_CMD) nextest run

.PHONY: doc
doc: format ## Generate the documentation
	@echo "Generating documentation..."
	@$(CARGO_CMD) doc --no-deps --document-private-items

########################################################################################
## Python Targets
########################################################################################

.PHONY: develop-py
develop-py: ## Build and install PyVq in the current Python environment
	@echo "Building and installing PyVq..."
	# Note: Maturin does not work when both CONDA_PREFIX and VIRTUAL_ENV are set.
	@(cd $(PYVQ_DIR) && unset CONDA_PREFIX && maturin develop)

.PHONY: wheel
wheel: ## Build the wheel file for PyVq
	@echo "Building the PyVq wheel..."
	@(cd $(PYVQ_DIR) && maturin build --release --out $(WHEEL_DIR) --auditwheel check)

.PHONY: wheel-manylinux
wheel-manylinux: ## Build the manylinux wheel file for PyVq (using Zig)
	@echo "Building the manylinux PyVq wheel..."
	@(cd $(PYVQ_DIR) && maturin build --release --out $(WHEEL_DIR) --auditwheel check --zig)

.PHONY: test-py
test-py: develop-py ## Run Python tests
	@echo "Running Python tests..."
	@poetry run pytest $(PYVQ_DIR)/tests

.PHONY: publish-py
publish-py: wheel-manylinux ## Publish the PyVq wheel to PyPI (requires PYPI_TOKEN to be set)
	@echo "Publishing PyVq to PyPI..."
	@if [ -z "$(WHEEL_FILE)" ]; then \
	  echo "Error: No wheel file found. Please run 'make wheel' first."; \
	  exit 1; \
	fi
	@echo "Found wheel file: $(WHEEL_FILE)"
	@twine upload -u __token__ -p $(PYPI_TOKEN) $(WHEEL_FILE)

.PHONY: generate-ci
generate-ci: ## Generate CI configuration files (GitHub Actions workflow)
	@echo "Generating CI configuration files..."
	@(cd $(PYVQ_DIR) && maturin generate-ci --zig --pytest --platform all -o ../.github/workflows/ci.yml github)
