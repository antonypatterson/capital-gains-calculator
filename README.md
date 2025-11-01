# capital-gains-calculator

Targeted towards Australia-centric tax methods. A way to try and reduce the tediousness of calculating capital gains within excel by working within a more versatile package (i.e pandas in python) to acheive.

Still a work in progress but one day hope to make this a fully fledged application.

Future Improvements:
- Data should be stored into a `data/` folder
- Scripts into `scripts/` folder
- Class-based and function-only scripts into a `modules/` folder
- Data read/write abstracted into a separate module (within `modules/`) e.g. `datastore.py`
- Currently only AUD-native or AUD/USD conversion is permitted, so utilising all CCY pairs offered by the [RBA FX Rates Page](https://www.rba.gov.au/statistics/historical-data.html) would be the logical next step. This will open up the ability to utilise this app for international investment CGT analysis.
- Automate the scraping of the [RBA FX Rates Page](https://www.rba.gov.au/statistics/historical-data.html) so that the .xlsx exports don't need to be manually downloaded and hard-coded into the `cgt_calculator.py` file.

