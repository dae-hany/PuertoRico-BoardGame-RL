# puerto_rico_constants.py

# Game Configuration (2 Players)
NUM_PLAYERS = 2
INITIAL_DOUBLOONS = 3
INITIAL_COLONISTS_MARKET = 2
INITIAL_COLONISTS_SUPPLY = 40
INITIAL_VP_CHIPS = 65

# Goods
CORN = 0
FRUIT = 1  # Indigo in standard, Fruit (과일) in this rulebook
SUGAR = 2
TOBACCO = 3
COFFEE = 4
NUM_GOODS = 5

GOODS_NAMES = ["Corn", "Fruit", "Sugar", "Tobacco", "Coffee"]

# Supply Counts
GOODS_SUPPLY = [8, 9, 9, 7, 7]  # Corn, Fruit, Sugar, Tobacco, Coffee (Rulebook order: Corn 8, Fruit 9, Sugar 9, Tobacco 7, Coffee 7)

# Roles
SETTLER = 0    # 개척자
MAYOR = 1      # 모집관 
BUILDER = 2    # 건축가
CRAFTSMAN = 3  # 생산자
TRADER = 4     # 상인
CAPTAIN = 5    # 선장
PROSPECTOR = 6 # 탐험가
NUM_ROLES = 7

ROLE_NAMES = ["Settler", "Mayor", "Builder", "Craftsman", "Trader", "Captain", "Prospector"]

# Plantations
PLANTATION_CORN = 0
PLANTATION_FRUIT = 1
PLANTATION_SUGAR = 2
PLANTATION_TOBACCO = 3
PLANTATION_COFFEE = 4
PLANTATION_QUARRY = 5
NUM_PLANTATION_TYPES = 6

# Initial Plantation Tokens (excluding Quarry)
PLANTATION_COUNTS = {
    PLANTATION_COFFEE: 5,
    PLANTATION_TOBACCO: 6,
    PLANTATION_CORN: 7,
    PLANTATION_SUGAR: 8,
    PLANTATION_FRUIT: 9,
}
QUARRY_COUNT = 5 # Separate stack

# Buildings
# ID: (Name, Cost, VP, MaxWorkers, QuarryLimit)
BUILDING_SMALL_FRUIT = 0            # 소형 과일 공장
BUILDING_SMALL_SUGAR = 1            # 소형 설탕 공장 
BUILDING_LARGE_FRUIT = 2            # 대형 과일 공장
BUILDING_LARGE_SUGAR = 3            # 대형 설탕 공장
BUILDING_TOBACCO = 4                # 담배 공장
BUILDING_COFFEE = 5                 # 커피 공장

BUILDING_SMALL_MARKET = 6           # 소형 상가
BUILDING_HACIENDA = 7               # 농장
BUILDING_CONSTRUCTION_HUT = 8       # 건설막
BUILDING_SMALL_WAREHOUSE = 9        # 소형 창고
BUILDING_HOSPICE = 10               # 병원
BUILDING_OFFICE = 11                # 영업소
BUILDING_LARGE_MARKET = 12          # 대형 상가
BUILDING_LARGE_WAREHOUSE = 13       # 대형 창고
BUILDING_FACTORY = 14               # 공업소
BUILDING_UNIVERSITY = 15            # 학교
BUILDING_HARBOR = 16                # 항구
BUILDING_WHARF = 17                 # 조선소

# Large Buildings
BUILDING_GUILD_HALL = 18            # 소방서
BUILDING_RESIDENCE = 19             # 주거지
BUILDING_FORTRESS = 20              # 요새
BUILDING_CUSTOMS_HOUSE = 21         # 세관
BUILDING_CITY_HALL = 22             # 시청

NUM_BUILDINGS = 23

# Building Specs
# (Cost, VP, Workers, QuarryLimit)
BUILDING_INFO = {
    BUILDING_SMALL_FRUIT:      (1, 1, 1, 1),
    BUILDING_SMALL_SUGAR:      (2, 1, 1, 1),
    BUILDING_LARGE_FRUIT:      (3, 2, 3, 2),
    BUILDING_LARGE_SUGAR:      (4, 2, 3, 2),
    BUILDING_TOBACCO:          (5, 3, 3, 3),
    BUILDING_COFFEE:           (6, 3, 2, 3),
    
    BUILDING_SMALL_MARKET:     (1, 1, 1, 1), 
    BUILDING_HACIENDA:         (2, 1, 1, 1), 
    BUILDING_CONSTRUCTION_HUT: (2, 1, 1, 1), 
    BUILDING_SMALL_WAREHOUSE:  (3, 1, 1, 1), 
    BUILDING_HOSPICE:          (4, 2, 1, 2), 
    BUILDING_OFFICE:           (5, 2, 1, 2), 
    BUILDING_LARGE_MARKET:     (5, 2, 1, 2), 
    BUILDING_LARGE_WAREHOUSE:  (6, 2, 1, 2),
    BUILDING_FACTORY:          (7, 3, 1, 3), 
    BUILDING_UNIVERSITY:       (8, 3, 1, 3), 
    BUILDING_HARBOR:           (8, 3, 1, 3), 
    BUILDING_WHARF:            (9, 3, 1, 3), 
    
    BUILDING_GUILD_HALL:       (10, 4, 1, 4), 
    BUILDING_RESIDENCE:        (10, 4, 1, 4), 
    BUILDING_FORTRESS:         (10, 4, 1, 4), 
    BUILDING_CUSTOMS_HOUSE:    (10, 4, 1, 4), 
    BUILDING_CITY_HALL:        (10, 4, 1, 4), 
}

# Building Counts for 2 Players(생산 건물은 2개씩, 상업 건물과 고급 건물은 1개씩)
BUILDING_COUNTS = {i: 2 for i in range(6)} 
for i in range(6, NUM_BUILDINGS):
    BUILDING_COUNTS[i] = 1 

# Ships
SHIP_CAPACITIES = [4, 6]

# Max limits for scaling/normalization (Observation Space)
MAX_DOUBLOONS_OBS = 20  # Soft cap for obs normalization if needed
MAX_VP_OBS = 100
MAX_GOODS_OBS = 12      # Can have more but rare

# Action Mapping
ACTION_CHOOSE_ROLE_SETTLER = 0
ACTION_CHOOSE_ROLE_MAYOR = 1
ACTION_CHOOSE_ROLE_BUILDER = 2
ACTION_CHOOSE_ROLE_CRAFTSMAN = 3
ACTION_CHOOSE_ROLE_TRADER = 4
ACTION_CHOOSE_ROLE_CAPTAIN = 5
ACTION_CHOOSE_ROLE_PROSPECTOR = 6

ACTION_SETTLER_TAKE_PLANTATION_0 = 7
ACTION_SETTLER_TAKE_PLANTATION_1 = 8
ACTION_SETTLER_TAKE_PLANTATION_2 = 9
ACTION_SETTLER_TAKE_QUARRY = 10

ACTION_BUILD_START = 11
# Mapping: Action = ACTION_BUILD_START + BuildingID (0-22) -> 11-33
"""여기까지 읽었음"""
ACTION_SELL_CORN = 34
ACTION_SELL_FRUIT = 35
ACTION_SELL_SUGAR = 36
ACTION_SELL_TOBACCO = 37
ACTION_SELL_COFFEE = 38

ACTION_SHIP_CORN = 39
ACTION_SHIP_FRUIT = 40
ACTION_SHIP_SUGAR = 41
ACTION_SHIP_TOBACCO = 42
ACTION_SHIP_COFFEE = 43

ACTION_CRAFTSMAN_BONUS_CORN = 44
ACTION_CRAFTSMAN_BONUS_FRUIT = 45
ACTION_CRAFTSMAN_BONUS_SUGAR = 46
ACTION_CRAFTSMAN_BONUS_TOBACCO = 47
ACTION_CRAFTSMAN_BONUS_COFFEE = 48

# Mayor: Place colonist on Plantation Slot 0-11
ACTION_MAYOR_PLACE_PLANTATION_0 = 49
# ... +11
ACTION_MAYOR_PLACE_PLANTATION_11 = 60

# Mayor: Place colonist on Building Slot 0-11
ACTION_MAYOR_PLACE_BUILDING_0 = 61
# ... +11
ACTION_MAYOR_PLACE_BUILDING_11 = 72

ACTION_PASS = 73

NUM_ACTIONS = 74

# Phases
PHASE_ROLE_SELECTION = 0
PHASE_SETTLER = 1
PHASE_MAYOR = 2
PHASE_BUILDER = 3
PHASE_CRAFTSMAN = 4
PHASE_TRADER = 5
PHASE_CAPTAIN = 6
PHASE_PROSPECTOR = 7
PHASE_GAME_END = 99