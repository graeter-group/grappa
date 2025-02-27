# %%
"""
Showcase the Grappa contributions to a dataset.

The Grappa contribution are printed-out and illustrated in a barplot.
Note that the dataset 'spice-dipeptide-amber99-residue-grappa-test' needs to be created beforehand by running the script 'create_spice_dipetide_partial.py' and subsequently 'create_test_dataset.py'.
"""
import matplotlib.pyplot as plt
from grappa.utils.graph_utils import get_grappa_contributions

# %%
# Calculate the Grappa contributions
contributions = get_grappa_contributions('spice-dipeptide-amber99-residue-grappa-test')

# %%
# Print the grappa contributions
for key, value in contributions.items():
    print(f"{key}: {value}")

# %%
# Plot the ratio of Grappa interactions in a barplot
ax = plt.bar(["Bonds", "Angles", "Propers", "Impropers"], [contributions["bonds_contrib"], contributions["angles_contrib"], contributions["propers_contrib"], contributions["impropers_contrib"]], edgecolor="black", color=["#4878d0", "#6acc64", "#ee854a", "#dc7ec0"])
plt.ylabel("Ratio of Grappa interactions", size=12)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(0, 1)
plt.show()
# %%
