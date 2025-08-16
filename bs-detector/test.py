from dax_bs_detector import compute_vagueness_score
text = "We are determined to end poverty and h‌ unger in all their‌ forms and dimensions, and to ensure that all human beings can fulfill their potential in dignity and equality and in a‌ healthy environment.‌"
r = compute_vagueness_score(text, threshold=0.3)
print(r.score, r.label, r.features)
