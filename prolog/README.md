## SICStus Prolog Installation (Linux)

1. Download the SICStus Prolog tar.gz file from the official site.
2. Extract and install using:
	```bash
	cat <downloaded-file>.tar.gz | gzip -cd | tar xf -
	cd sp-<version>-<platform>
	sudo ./InstallSICStus
	```
3. Check your license information at: [https://moodle2526.up.pt/mod/page/view.php?id=108772](https://moodle2526.up.pt/mod/page/view.php?id=108772)

---

### Setting Up Environment Variables

To set up the environment variables for SICStus Prolog, add the following lines to your `~/.bashrc` or `~/.bash_profile` file:

```bash
echo 'export PATH=$PATH:/usr/local/sicstus4.10.1/bin' >> ~/.bashrc
source ~/.bashrc
```

Installation of rl_wrap may be required for command line editing features:

```bash
sudo apt-get install rlwrap
```

Start rlwrap with SICStus Prolog:

```bash
rlwrap sicstus
```

## Quick Prolog Tutorial

### What is Prolog?
Prolog is a logic programming language used for solving problems involving objects and relationships between them. It is based on facts, rules, and queries.

### Basic Concepts
- **Facts:** State things that are true.
  ```prolog
  parent(john, mary).
  parent(mary, susan).
  ```
- **Rules:** Define relationships using facts.
  ```prolog
  grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
  ```
- **Queries:** Ask questions about facts/rules.
  ```prolog
  ?- grandparent(john, susan).
  ```

### How to Run a Prolog Program
1. Start SICStus Prolog by typing `sicstus` in your terminal.
2. Load your Prolog file:
	```prolog
	?- [filename].
	```
	(Replace `filename` with your `.pl` file, without the extension.)
3. Run queries:
	```prolog
	?- parent(john, mary).
	```
4. Exit Prolog:
	```prolog
	?- halt.
	```

---

For more details, see the official SICStus Prolog documentation.
