
Test that the TICG simulation contact map coarse graining (binning) matches the 
epilib contact map pooling.

two options:
conservative pooling conserves the total number of contacts in the map,
but as a consequence treats the main diagonal differently

nonconservative pooling applies a sumpool operation across the entire
contactmap matrix, but as a result does not conserve total number
of contacts

