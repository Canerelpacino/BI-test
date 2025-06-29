from typing import List, Dict, Set, Optional, Tuple
def calculate_recall_and_positions(reference_items: List[Dict],
                                   predicted_items: List[Dict],
                                   k: int = 5,
                                   use_location: bool = True,
                                   use_year: bool = True,
                                   use_page: bool = True) -> Tuple[float, List[int]]:
    """
    Calculate recall@k and positions of matches in predicted items.

    Args:
        reference_items: List of reference items with location, year, and page
        predicted_items: List of predicted items to compare against reference
        k: Number of top predictions to consider (default=5)
        use_location: Consider location in matching (default=True)
        use_year: Consider year in matching (default=True)
        use_page: Consider page in matching (default=True)

    Returns:
        Tuple[float, List[int]]: (Recall@k value between 0 and 1, List of match positions)
    """
    if not reference_items:
        print("No reference items found")
        return 1.0, []

    relevant_items: Set = set()
    match_positions: List[int] = []

    # Create unique identifiers for reference items
    for ref in reference_items:
        identifier = []
        if use_location and 'ort' in ref:
            identifier.append(str(ref['ort']))
        if use_year and 'jahr' in ref:
            identifier.append(str(ref['jahr']))
        if use_page and 'page' in ref:
            identifier.append(str(ref['page']))
        if identifier:
            relevant_items.add(tuple(identifier))


    # Check predictions against reference set and record positions
    for idx, pred in enumerate(predicted_items[:k]):
        identifier = []
        if use_location and 'ort' in pred:
            identifier.append(str(pred['ort']))
        if use_year and 'jahr' in pred:
            identifier.append(str(pred['jahr']))
        if use_page and 'page' in pred:
            identifier.append(str(pred['page']))
        if identifier:
            pred_tuple = tuple(identifier)
            if pred_tuple in relevant_items:
                match_positions.append(idx)

    recall = len(match_positions) / len(relevant_items) if relevant_items else 1.0
    return recall, match_positions