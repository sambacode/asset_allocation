EM_CDS_TRACKER_DICT = {
    "AED": "GSCDABBE Index", # no FX Tracker
    "ARS": "GSCDARBE Index",
    "BRL": "GSCDBRBE Index",
    "CLP": "GSCDCLBE Index",
    "CNY": "GSCDCHBE Index",
    "COP": "GSCDCOBE Index",
    "IDR": "GSCDINBE Index",
    "MYR": "GSCDMABE Index",
    "MXN": "GSCDMEBE Index",
    "PAB": "GSCDPABE Index",
    "PEN": "GSCDPEBE Index",
    "PHP": "GSCDPHBE Index",
    "QAR": "GSCDQABE Index",
    "RUB": "GSCDRUBE Index",
    "SAR": "GSCDSABE Index",
    "ZAR": "GSCDSOBE Index",
    "TRY": "GSCDTUBE Index",
    "UAH": "GSCDUKBE Index",
    # "CDX EM": "EREM5LD5 Index", # no FX Tracker
}


IRS_TRACKER_DICT = {
    "BRL": "GSSWBRN5 Index",
    "CNY": "GSSWCNN5 Index",
    "MXN": "GSSWMXN5 Index",
    "ZAR": "GSSWZAN5 Index",
}

EQ_TRACKER_DICT = {
    "BRL": "BNPIFBR Index",  # in BRL
    "CNY": "BNPIFCNO Index",  # China onshore but with pnl converted to USD
    "ZAR": "BNPIFSA Index",  # in ZAR
    # "MXN": "???? Index",
}
fx_trackers_aa = [
    "BRL",
    "CLP",
    "CNH",
    "CNY",
    "CZK",
    "HUF",
    "IDR",
    "INR",
    "MXN",
    "PLN",
    "RUB",
    "SEK",
    "SGD",
    "TRY",
    "TWD",
    "ZAR",
    "THB",
    "COP",
    "MYR",
    "HKD",
]
FX_TRACKER_DICT = {
    "ARS": "JPFCTARS Index",
    "BRL": "JPFCTBRL Index",
    "CLP": "JPFCTCLP Index",
    "CNY": "JPFCTCNY Index",
    "COP": "JPFCTCOP Index",
    "CZK": "JPFCTCZK Index",  # no CDS tracker
    "HUF": "JPFCTHUF Index",  # no CDS tracker
    "IDR": "JPFCTIDR Index",
    "INR": "JPFCTINR Index",  # no CDS tracker
    "MXN": "JPFCTMXN Index",
    "MYR": "JPFCTMYR Index",
    # "PAB": "JPFCTPAB Index",  # pegged to dollar?
    "PEN": "JPFCTPEN Index",
    "PHP": "JPFCTPHP Index",
    "PLN": "JPFCTPLN Index", # no CDS tracker
    "QAR": "JPFCTQAR Index",
    "RON": "JPFCTRON Index",  # no CDS tracker
    "RUB": "JPFCTRUB Index",
    # "SAR": "JPFCTSAR Index",  # pegged to dollar?
    "SGD": "JPFCTSGD Index",  # no CDS tracker
    "THB": "JPFCTTHB Index",  # no CDS tracker
    "TRY": "JPFCTTRY Index",
    "TWD": "JPFCTTWD Index",  # no CDS tracker
    "UAH": "JPFCTUAH Index",
    "ZAR": "JPFCTZAR Index",
}
