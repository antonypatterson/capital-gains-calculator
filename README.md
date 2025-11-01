# capital-gains-calculator

Targeted towards Australia-centric tax methods. A way to try and reduce the tediousness of calculating capital gains within excel by working within a more versatile package (i.e pandas in python) to acheive.

Still a work in progress but one day hope to make this a fully fledged application. This is merely a hobby project at the moment, but would like to have this as a way to distribute a more robust CGT calculation to everyday punters who have basic script-execution experience, without necessarily needing to peep into the underlying code. Although there are websites that offer a fully fledged service, they are either restrictive with their "free" offerings, or expensive for their "premium" plans. 

One alternative is to do it yourself via excel, but this becomes incredibly difficult when needing to consider which method of calcuation to apply (where the most common is first in first out i.e. FIFO). With small volumes of trades, this can be done manually. However, as people are moving more and more towards fractional investing at regular intervals, there may be hundreds or thousands of transaction records in a given tax year.

Future Improvements:
- Data should be stored into a `data/` folder
- Scripts into `scripts/` folder
- Class-based and function-only scripts into a `modules/` folder
- Data read/write abstracted into a separate module (within `modules/`) e.g. `datastore.py`
- Currently only AUD-native or AUD/USD conversion is permitted, so utilising all CCY pairs offered by the [RBA FX Rates Page](https://www.rba.gov.au/statistics/historical-data.html) would be the logical next step. This will open up the ability to utilise this app for international investment CGT analysis.
- Automate the scraping of the [RBA FX Rates Page](https://www.rba.gov.au/statistics/historical-data.html) so that the .xlsx exports don't need to be manually downloaded and hard-coded into the `cgt_calculator.py` file.

