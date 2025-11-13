from copy import copy

def find_headings(lines: list[str]) -> tuple[list[str], list[list[int]], list[str]]:
    """
    Finds all headings from a list of lines
    :param lines: list[str]
    :return: tuple[list[str], list[list[int]]
    """
    sm_titles = []  # The names of the titles
    title_indices = []
    ordered_lines = copy(lines)
    # TODO include the first catalogue entry as well
    for i, l in enumerate(lines):
        sm = find_shelfmark(l)
        if sm: 
            title = [l]
            title_index = []
            j = 1
            while i + j < len(lines) and j < 8:
                title_part = lines[i + j]
                if find_shelfmark(title_part):  # If a new catalogue entry begins during the current title
                    break

                title.append(title_part)
                title_index.append(i + j)
                j += 1

                if date_check(title_part) and caps_regex.search(" ". join(title)):  # Date marks the end of a heading
                    sm_titles.append([sm, title])
                    if "Bought in" in title[1]:  # not .lower() - these "Bought in" should all be capitalised
                        sm, bought_in = lines[i], lines[i+1]
                        ordered_lines[i], ordered_lines[i+1] = bought_in, sm
                        title_indices.append(title_index[1:])
                    else:
                        title_indices.append(title_index)
                    break

    title_shelfmarks = [t[0] for t in sm_titles]

    return title_shelfmarks, title_indices, ordered_lines


print("hello world")