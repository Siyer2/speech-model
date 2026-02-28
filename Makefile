.PHONY: install train test lint format typecheck quality pre-commit \
       model-% frontend-%

# Delegate to model/
install train test lint format typecheck quality pre-commit:
	$(MAKE) -C model $@ EVAL_ONLY="$(EVAL_ONLY)" NOTE="$(NOTE)" NAME="$(NAME)"

# Explicit sub-project targets: `make model-train`, `make frontend-install`, etc.
model-%:
	$(MAKE) -C model $*

frontend-%:
	$(MAKE) -C frontend $*
