"""
COT Contract Configuration — all 37 markets tracked by the CMR (Crowded Market Report).
Single source of truth for CFTC name matching, Yahoo Finance tickers, and sector mapping.

Each contract has a 'codes' list of CFTC Contract Market Codes for reliable matching
(CFTC occasionally renames markets but codes stay stable). The 'search' list is a
fallback for name-based matching.
"""

COT_CONTRACTS = {
    # ── Equities (4) ──────────────────────────────────────────────────────
    'sp500':   {'label': 'S&P 500 E-mini',  'sector': 'Equities',     'codes': ['13874A'],  'search': ['E-MINI S&P 500 STOCK INDEX', 'E-MINI S&P 500'],  'yf': 'ES=F', 'equity': True},
    'nasdaq':  {'label': 'Nasdaq 100',       'sector': 'Equities',     'codes': ['209742'],  'search': ['NASDAQ-100 STOCK INDEX (MINI)'],                 'yf': 'NQ=F', 'equity': True},
    'dow':     {'label': 'Dow Jones',        'sector': 'Equities',     'codes': ['124603'],  'search': ['DOW JONES INDUSTRIAL AVG- x $5'],                'yf': 'YM=F', 'equity': True},
    'russell': {'label': 'Russell 2000',     'sector': 'Equities',     'codes': ['239742'],  'search': ['RUSSELL 2000 MINI INDEX', 'RUSSELL E-MINI'],     'yf': 'RTY=F','equity': True},

    # ── Fixed Income (4) ──────────────────────────────────────────────────
    'bond30':  {'label': '30Y T-Bond',       'sector': 'Fixed Income', 'codes': ['020601'],  'search': ['U.S. TREASURY BONDS - CHICAGO'],                 'yf': 'ZB=F', 'equity': False},
    'tnote10': {'label': '10Y T-Note',       'sector': 'Fixed Income', 'codes': ['043602'],  'search': ['10-YEAR U.S. TREASURY NOTES'],                   'yf': 'ZN=F', 'equity': False},
    'tnote5':  {'label': '5Y T-Note',        'sector': 'Fixed Income', 'codes': ['044601'],  'search': ['5-YEAR U.S. TREASURY NOTES'],                    'yf': 'ZF=F', 'equity': False},
    'tnote2':  {'label': '2Y T-Note',        'sector': 'Fixed Income', 'codes': ['042601'],  'search': ['2-YEAR U.S. TREASURY NOTES'],                    'yf': 'ZT=F', 'equity': False},

    # ── Currencies (7) ────────────────────────────────────────────────────
    'usdx':    {'label': 'US Dollar Index',  'sector': 'Currencies',   'codes': ['098662'],  'search': ['U.S. DOLLAR INDEX'],                             'yf': 'DX=F', 'equity': False},
    'aud':     {'label': 'Australian Dollar', 'sector': 'Currencies',  'codes': ['232741'],  'search': ['AUSTRALIAN DOLLAR - CHICAGO'],                   'yf': '6A=F', 'equity': False},
    'gbp':     {'label': 'British Pound',    'sector': 'Currencies',   'codes': ['096742'],  'search': ['BRITISH POUND STERLING', 'BRITISH POUND - CHICAGO'], 'yf': '6B=F', 'equity': False},
    'cad':     {'label': 'Canadian Dollar',  'sector': 'Currencies',   'codes': ['090741'],  'search': ['CANADIAN DOLLAR - CHICAGO'],                     'yf': '6C=F', 'equity': False},
    'jpy':     {'label': 'Japanese Yen',     'sector': 'Currencies',   'codes': ['097741'],  'search': ['JAPANESE YEN - CHICAGO'],                        'yf': '6J=F', 'equity': False},
    'eur':     {'label': 'Euro FX',          'sector': 'Currencies',   'codes': ['099741'],  'search': ['EURO FX - CHICAGO'],                             'yf': '6E=F', 'equity': False},
    'chf':     {'label': 'Swiss Franc',      'sector': 'Currencies',   'codes': ['092741'],  'search': ['SWISS FRANC - CHICAGO'],                         'yf': '6S=F', 'equity': False},

    # ── Metals (5) ────────────────────────────────────────────────────────
    'gold':      {'label': 'Gold',           'sector': 'Metals',       'codes': ['088691'],  'search': ['GOLD - COMMODITY EXCHANGE'],                     'yf': 'GC=F', 'equity': False},
    'silver':    {'label': 'Silver',         'sector': 'Metals',       'codes': ['084691'],  'search': ['SILVER - COMMODITY EXCHANGE'],                   'yf': 'SI=F', 'equity': False},
    'palladium': {'label': 'Palladium',      'sector': 'Metals',       'codes': ['075651'],  'search': ['PALLADIUM - NEW YORK'],                          'yf': 'PA=F', 'equity': False},
    'platinum':  {'label': 'Platinum',       'sector': 'Metals',       'codes': ['076651'],  'search': ['PLATINUM - NEW YORK'],                           'yf': 'PL=F', 'equity': False},
    'copper':    {'label': 'Copper',         'sector': 'Metals',       'codes': ['085692'],  'search': ['COPPER-GRADE #1'],                               'yf': 'HG=F', 'equity': False},

    # ── Energies (4) ──────────────────────────────────────────────────────
    'crude':       {'label': 'Crude Oil WTI',   'sector': 'Energies',  'codes': ['067651'],  'search': ['CRUDE OIL, LIGHT SWEET - NEW YORK'],             'yf': 'CL=F', 'equity': False},
    'gasoline':    {'label': 'RBOB Gasoline',   'sector': 'Energies',  'codes': ['111659'],  'search': ['GASOLINE BLENDSTOCK (RBOB)'],                    'yf': 'RB=F', 'equity': False},
    'heating_oil': {'label': 'Heating Oil',     'sector': 'Energies',  'codes': ['022651'],  'search': ['NO. 2 HEATING OIL, N.Y. HARBOR'],                'yf': 'HO=F', 'equity': False},
    'natgas':      {'label': 'Natural Gas',     'sector': 'Energies',  'codes': ['023651'],  'search': ['NATURAL GAS - NEW YORK'],                        'yf': 'NG=F', 'equity': False},

    # ── Softs (6) ─────────────────────────────────────────────────────────
    'cocoa':   {'label': 'Cocoa',            'sector': 'Softs',        'codes': ['073732'],  'search': ['COCOA - ICE FUTURES', 'COCOA - NEW YORK'],       'yf': 'CC=F', 'equity': False},
    'sugar':   {'label': 'Sugar #11',        'sector': 'Softs',        'codes': ['080732'],  'search': ['SUGAR NO. 11'],                                  'yf': 'SB=F', 'equity': False},
    'oj':      {'label': 'Orange Juice',     'sector': 'Softs',        'codes': ['040701'],  'search': ['FRZN CONCENTRATED ORANGE JUICE'],                'yf': 'OJ=F', 'equity': False},
    'coffee':  {'label': 'Coffee C',         'sector': 'Softs',        'codes': ['083731'],  'search': ['COFFEE C'],                                      'yf': 'KC=F', 'equity': False},
    'cotton':  {'label': 'Cotton #2',        'sector': 'Softs',        'codes': ['033661'],  'search': ['COTTON NO. 2'],                                  'yf': 'CT=F', 'equity': False},
    'lumber':  {'label': 'Lumber',           'sector': 'Softs',        'codes': ['058643'],  'search': ['RANDOM LENGTH LUMBER'],                          'yf': 'LBS=F','equity': False},

    # ── Grains (6) ────────────────────────────────────────────────────────
    'corn':         {'label': 'Corn',           'sector': 'Grains',    'codes': ['002602'],  'search': ['CORN - CHICAGO BOARD'],                          'yf': 'ZC=F', 'equity': False},
    'oats':         {'label': 'Oats',           'sector': 'Grains',    'codes': ['004603'],  'search': ['OATS - CHICAGO BOARD'],                          'yf': 'ZO=F', 'equity': False},
    'wheat':        {'label': 'Wheat (SRW)',    'sector': 'Grains',    'codes': ['001602'],  'search': ['WHEAT-SRW - CHICAGO', 'WHEAT - CHICAGO BOARD OF TRADE'], 'yf': 'ZW=F', 'equity': False},
    'soybeans':     {'label': 'Soybeans',       'sector': 'Grains',    'codes': ['005602'],  'search': ['SOYBEANS - CHICAGO BOARD'],                      'yf': 'ZS=F', 'equity': False},
    'soybean_meal': {'label': 'Soybean Meal',   'sector': 'Grains',   'codes': ['026603'],  'search': ['SOYBEAN MEAL - CHICAGO'],                        'yf': 'ZM=F', 'equity': False},
    'soybean_oil':  {'label': 'Soybean Oil',    'sector': 'Grains',   'codes': ['007601'],  'search': ['SOYBEAN OIL - CHICAGO'],                         'yf': 'ZL=F', 'equity': False},

    # ── Livestock (2) ─────────────────────────────────────────────────────
    'cattle': {'label': 'Live Cattle',       'sector': 'Livestock',    'codes': ['057642'],  'search': ['LIVE CATTLE - CHICAGO'],                         'yf': 'LE=F', 'equity': False},
    'hogs':   {'label': 'Lean Hogs',         'sector': 'Livestock',    'codes': ['054642'],  'search': ['LEAN HOGS - CHICAGO'],                           'yf': 'HE=F', 'equity': False},

    # ── Crypto (1) ────────────────────────────────────────────────────────
    'bitcoin': {'label': 'Bitcoin',          'sector': 'Crypto',       'codes': ['133741'],  'search': ['BITCOIN - CHICAGO MERCANTILE'],                  'yf': 'BTC-USD', 'equity': False},
}

SECTORS = [
    'Equities', 'Fixed Income', 'Currencies', 'Metals',
    'Energies', 'Softs', 'Grains', 'Livestock', 'Crypto',
]

# Sector display colors (Obsidian Glass theme)
SECTOR_COLORS = {
    'Equities':     '#58a6ff',
    'Fixed Income': '#bc8cff',
    'Currencies':   '#7bf2da',
    'Metals':       '#e3b341',
    'Energies':     '#ff6b6b',
    'Softs':        '#f78166',
    'Grains':       '#3fb950',
    'Livestock':    '#d2a8ff',
    'Crypto':       '#ffa657',
}
