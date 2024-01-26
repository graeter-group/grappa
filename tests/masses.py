#%%
def parse_atomic_masses(table):
    atomic_masses = {}

    for line in table.splitlines():
        parts = line.split()
        if parts:
            atomic_number = int(parts[0])
            mass = parts[3]
            # Remove any parentheses and their contents
            mass = ''.join(mass.split('(')[0])
            # remove any brackets but not their content:
            mass = mass.replace('[', '').replace(']', '')

            atomic_masses[atomic_number] = float(mass)
                
    return atomic_masses

# from https://iupac.qmul.ac.uk/AtWt/

table = r'''1	H	Hydrogen	1.008	3, 5
2	He	Helium	4.002 602(2)	1, 2
3	Li	Lithium	6.94	3, 5
4	Be	Beryllium	9.012 1831(5)
5	B	Boron	10.81	3, 5
6	C	Carbon	12.011	5
7	N	Nitrogen	14.007	5
8	O	Oxygen	15.999	5
9	F	Fluorine	18.998 403 163(5)
10	Ne	Neon	20.1797(6)	1, 3
11	Na	Sodium	22.989 769 28(2)    
12	Mg	Magnesium	24.305	5
13	Al	Aluminium	26.981 5384(3)
14	Si	Silicon	28.085	5
15	P	Phosphorus	30.973 761 998(5)
16	S	Sulfur	32.06	5
17	Cl	Chlorine	35.45	3, 5
18	Ar	Argon	39.95	1, 2, 5
19	K	Potassium	39.0983(1)	
20	Ca	Calcium	40.078(4)	
21	Sc	Scandium	44.955 907(4)
22	Ti	Titanium	47.867(1)
23	V	Vanadium	50.9415(1)
24	Cr	Chromium	51.9961(6)
25	Mn	Manganese	54.938 043(2)
26	Fe	Iron	55.845(2)
27	Co	Cobalt	58.933 194(3)
28	Ni	Nickel	58.6934(4)	2
29	Cu	Copper	63.546(3)	2
30	Zn	Zinc	65.38(2)	2
31	Ga	Gallium	69.723(1)
32	Ge	Germanium	72.630(8)
33	As	Arsenic	74.921 595(6)
34	Se	Selenium	78.971(8)
35	Br	Bromine	79.904	5
36	Kr	Krypton	83.798(2)	1, 3
37	Rb	Rubidium	85.4678(3)	1
38	Sr	Strontium	87.62(1)	1, 2
39	Y	Yttrium	88.905 838(2)
40	Zr	Zirconium	91.224(2)	1
41	Nb	Niobium	92.906 37(1)
42	Mo	Molybdenum	95.95(1)	1
43	Tc	Technetium	[97]	4
44	Ru	Ruthenium	101.07(2)	1
45	Rh	Rhodium	102.905 49(2)
46	Pd	Palladium	106.42(1)	1
47	Ag	Silver	107.8682(2)	1
48	Cd	Cadmium	112.414(4)	1
49	In	Indium	114.818(1)
50	Sn	Tin	118.710(7)	1
51	Sb	Antimony	121.760(1)	1
52	Te	Tellurium	127.60(3)	1
53	I	Iodine	126.904 47(3)'''

atomic_masses = parse_atomic_masses(table)
atomic_masses

# %%
