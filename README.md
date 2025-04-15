IMPORTANT (setup):
1. In `modules/data`, unzip the `annotations.json.zip` file
2. Ensure the unzipped `annotations.json` file is in the `modules/data` folder
3. In the root directory, make a copy of `config.ini.sample` called `config.ini`, and put in your OpenAI API key
4. Switch client in config.ini to `gpt-4o` for better results.

To run the app:
1. `cd` into the root directory of the repo
2. Run the script with `python app.py`

To test modules on their own:
1. `cd` into the root directory
2. Run `python -m modules.<name_of_module>` (example: `python -m modules.rag`)

If a script takes too long to run, CTRL-C out and re-run the script.