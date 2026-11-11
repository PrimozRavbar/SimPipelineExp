"""Registration data as model inputs: age bracket, gender, occupation.

The ML system legitimately has these (they come with the account), and in
MovieLens they correlate with taste. Encoded as a fixed (not learned) table
indexed by user id: 7 age brackets + gender + one-hot occupation.
"""

AGE_BRACKETS = (18, 25, 35, 45, 50, 56)     # <18, 18-24, ..., 56+


def demographic_vector(user, occupations):
    age = [0.0] * (len(AGE_BRACKETS) + 1)
    age[sum(user.age >= b for b in AGE_BRACKETS)] = 1.0
    occupation = [1.0 if user.occupation == o else 0.0 for o in occupations]
    return age + [1.0 if user.gender == "F" else 0.0] + occupation


def demographics_table(user_objects):
    """torch tensor (max user id + 1, D); row 0 and unknown ids are zero."""
    import torch

    occupations = sorted({u.occupation for u in user_objects.values()})
    width = len(AGE_BRACKETS) + 1 + 1 + len(occupations)
    table = torch.zeros(max(user_objects) + 1, width)
    for user_id, user in user_objects.items():
        table[user_id] = torch.tensor(demographic_vector(user, occupations))
    return table
