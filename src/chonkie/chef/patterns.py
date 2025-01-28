"""Common text patterns and abbreviations used across chefs."""

from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class Abbreviations:
    """Common abbreviations grouped by category."""
    
    # Titles and honorifics
    TITLES: Set[str] = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.',
        'Sr.', 'Jr.', 'Rev.', 'Hon.',
    }
    
    # Academic and professional 
    ACADEMIC: Set[str] = {
        'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'B.Sc.',
        'M.Sc.', 'D.Phil.', 'LL.B.', 'LL.M.',
    }
    
    # Latin abbreviations
    LATIN: Set[str] = {
        'etc.', 'e.g.', 'i.e.', 'viz.',
        'vs.', 'al.', 'et al.', 'cf.',
    }
    
    # Military and government
    MILITARY: Set[str] = {
        'Gen.', 'Col.', 'Lt.', 'Sgt.', 'Capt.',
        'Maj.', 'Adm.', 'Gov.', 'Sen.', 'Rep.',
    }
    
    # Common measurements
    MEASUREMENTS: Set[str] = {
        'cm.', 'mm.', 'km.', 'kg.', 'lb.',
        'ft.', 'in.', 'hr.', 'min.', 'sec.',
    }
    
    # Business and organization
    BUSINESS: Set[str] = {
        'Inc.', 'Ltd.', 'Corp.', 'Co.', 'LLC.',
        'dept.', 'div.', 'est.', 'avg.', 'approx.',
    }
    
    # Temporal abbreviations
    TEMPORAL: Set[str] = {
        'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.',
        'Jul.', 'Aug.', 'Sep.', 'Sept.', 'Oct.',
        'Nov.', 'Dec.', 'Mon.', 'Tue.', 'Wed.',
        'Thu.', 'Fri.', 'Sat.', 'Sun.',
    }
    
    # Geographical abbreviations
    GEOGRAPHICAL: Set[str] = {
        'U.S.', 'U.S.A.', 'U.K.', 'E.U.',
        'Ave.', 'Blvd.', 'Rd.', 'St.', 'Mt.',
    }

    @classmethod
    def all(cls) -> Set[str]:
        """Return all abbreviations as a single set."""
        all_abbrevs = set()
        for field_name in cls.__annotations__:
            if isinstance(field_name, str) and field_name.isupper():
                all_abbrevs.update(getattr(cls, field_name))
        return all_abbrevs

    @classmethod
    def by_category(cls) -> Dict[str, Set[str]]:
        """Return abbreviations grouped by category."""
        categories = {}
        for field_name in cls.__annotations__:
            if isinstance(field_name, str) and field_name.isupper():
                categories[field_name.lower()] = getattr(cls, field_name)
        return categories


@dataclass(frozen=True, slots=True)
class UnicodeReplacements:
    """Common Unicode characters used for replacements."""
    
    DOT_LEADER = '․'  # U+2024 ONE DOT LEADER
    ELLIPSIS = '…'    # U+2026 HORIZONTAL ELLIPSIS
