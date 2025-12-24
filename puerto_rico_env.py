
import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
import puerto_rico_constants as c

class GameState:
    def __init__(self):
        self.players = []
        self.supply_goods = list(c.GOODS_SUPPLY)
        self.supply_colonists = c.INITIAL_COLONISTS_SUPPLY
        self.supply_vp = c.INITIAL_VP_CHIPS
        self.supply_quarries = c.QUARRY_COUNT
        
        # Plantations
        self.plantation_deck = []
        self.market_plantations = [] # Face up
        self.discarded_plantations = []
        
        # Roles
        self.roles_available = [True] * c.NUM_ROLES
        self.roles_doubloons = [0] * c.NUM_ROLES
        
        # Trading House [GoodID or -1]
        self.trading_house = [-1] * 4
        
        # Ships [GoodID, Count, Capacity]
        # Ships capacities are 4 and 6 for 2 players
        self.ships = [
            {'good': -1, 'count': 0, 'capacity': 4},
            {'good': -1, 'count': 0, 'capacity': 6}
        ]
        
        self.governor_idx = 0
        self.current_player_idx = 0
        self.colonist_ship = c.INITIAL_COLONISTS_MARKET
        
        # Building supply
        self.building_supply = c.BUILDING_COUNTS.copy() # dict {id: count}
        
        # Turn/Phase Control
        self.phase = c.PHASE_ROLE_SELECTION
        self.roles_taken_count = 0 # 0 to 6 in a round
        self.current_role = -1
        self.current_role_privilege = False # Does current actor have privilege?
        
        # Who is acting right now?
        # In role phase: the player whose turn it is to pick.
        # In action phase: the player defined by the queue.
        self.action_queue = [] # List of player indices
        self.captain_consecutive_passes = 0 # Track passes in Captain Phase


class PlayerState:
    def __init__(self):
        self.doubloons = c.INITIAL_DOUBLOONS
        self.vp_chips = 0
        self.goods = [0] * c.NUM_GOODS
        
        # 12 Island Slots: List of {'tile': ID, 'workers': count}
        # In rulebook: "12칸의 토지"
        self.island = [{'tile': -1, 'workers': 0} for _ in range(12)]
        
        # 12 City Slots: List of {'building': ID, 'workers': count}
        # In rulebook: "12칸의 건설 부지"
        self.city = [{'building': -1, 'workers': 0} for _ in range(12)]
        
        self.san_juan_workers = 0 # "개인판 우측 상단" (San Juan / Windrose)
        self.last_produced_goods = [0] * c.NUM_GOODS # For Craftsman bonus tracking
        self.wharf_used = False # For Captain phase tracking

class PuertoRicoEnv2P(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        
        # Define Observation Space
        # Global State Vector:
        # 0: Colonist Supply
        # 1: VP Supply
        # 2: Quarry Supply
        # 3-7: Goods Supply (5)
        # 8-14: Role Available (7) - 1 if available, 0 if not
        # 15-21: Role Doubloons (7)
        # 22-25: Trading House (4) - Good ID or -1
        # 26-28: Ship 1 (Good, Count, Capacity)
        # 29-31: Ship 2 (Good, Count, Capacity)
        # 32: Governor Index
        # 33: Current Player Index
        # 34: Colonist Ship Count
        # Total: ~35
        self.global_space_dim = 35
        
        # Player State Vector (per player):
        # 0: Doubloons
        # 1: VP Chips
        # 2-6: Goods Held (5)
        # 7-30: Island (12 slots * 2 values: TileID, Workers) = 24
        # 31-54: City (12 slots * 2 values: BldgID, Workers) = 24
        # 55: San Juan Workers
        # Total: 56
        self.player_space_dim = 56
        
        # Market (3 face up plantations)
        # 3 Ints
        
        self.observation_space = spaces.Dict({
            "global": spaces.Box(low=-1, high=100, shape=(self.global_space_dim,), dtype=np.int32),
            "players": spaces.Box(low=-1, high=100, shape=(c.NUM_PLAYERS, self.player_space_dim), dtype=np.int32),
            "market_plantations": spaces.Box(low=0, high=c.NUM_PLANTATION_TYPES, shape=(3,), dtype=np.int32)
        })
        
        self.game_state = None
        self.action_space = spaces.Discrete(c.NUM_ACTIONS)

    def get_action_mask(self):
        mask = np.zeros(c.NUM_ACTIONS, dtype=np.int8)
        gs = self.game_state
        if gs is None:
            return mask
            
        # If queue is active, current player is determined by queue
        # But wait, self.game_state.current_player_idx needs to be kept in sync?
        # Yes, I should update current_player_idx at start of step if queue changes.
        
        current_p_idx = gs.current_player_idx
        current_p = gs.players[current_p_idx]
        
        if gs.phase == c.PHASE_ROLE_SELECTION:
            # Mask available roles
            for r_id in range(c.NUM_ROLES):
                if gs.roles_available[r_id]:
                    mask[c.ACTION_CHOOSE_ROLE_SETTLER + r_id] = 1
        
        elif gs.phase == c.PHASE_SETTLER:
            # Market plantations (Indexes 0, 1, 2)
            for i in range(len(gs.market_plantations)):
                mask[c.ACTION_SETTLER_TAKE_PLANTATION_0 + i] = 1
            
            # Construction Hut Check
            has_hut = False
            for slot in current_p.city:
                if slot['building'] == c.BUILDING_CONSTRUCTION_HUT and slot['workers'] > 0:
                    has_hut = True
                    break

            # Quarry: Only if privilege is active OR Construction Hut
            if (gs.current_role_privilege or has_hut) and gs.supply_quarries > 0:
                mask[c.ACTION_SETTLER_TAKE_QUARRY] = 1
            
            # Pass (Optional? Line 62: "Player can choose not to take")
            # Usually players must take if they can? 
            # Rulebook Line 62: "cf. Playuer is not required to take a plantation tile if they don't want to."
            # So always allow PASS or treat "Take Nothing" as action?
            # I'll enable PASS for Settler.
            pass_action = c.ACTION_PASS # Define this in constants? It is 73.
            mask[pass_action] = 1
                
        # Custom Logic for Mayor Placement Masking
        elif gs.phase == c.PHASE_MAYOR:
            can_place = False
            # If current player has stored colonists in San Juan, they can place them
            if current_p.san_juan_workers > 0:
                # 1. Place on Island
                for i, slot in enumerate(current_p.island):
                    # Valid if slot has tile and is empty
                    if slot['tile'] != -1 and slot['workers'] == 0:
                        mask[c.ACTION_MAYOR_PLACE_PLANTATION_0 + i] = 1
                        can_place = True
                        
                # 2. Place on City
                for i, slot in enumerate(current_p.city):
                    if slot['building'] != -1:
                        b_id = slot['building']
                        capacity = c.BUILDING_INFO[b_id][2]
                        if slot['workers'] < capacity:
                             mask[c.ACTION_MAYOR_PLACE_BUILDING_0 + i] = 1
                             can_place = True
                             
            # 3. Pass Logic
            # "Player cannot voluntarily store colonists... must place... if there is empty slot."
            # So Pass is allowed ONLY if you CANNOT place.
            # OR if san_juan_workers == 0 (You are done).
            
            if not can_place:
                mask[c.ACTION_PASS] = 1
            else:
                 mask[c.ACTION_PASS] = 0

        elif gs.phase == c.PHASE_TRADER:
            # Check Trading House Full?
            house_full = True
            for i in range(4):
                if gs.trading_house[i] == -1:
                    house_full = False
                    break
            
            can_sell = False
            if not house_full:
                 # Check active Office
                 has_office = False
                 for slot in current_p.city:
                     if slot['building'] == c.BUILDING_OFFICE and slot['workers'] > 0:
                         has_office = True
                         break
                         
                 # Check each good held
                 for g_id in range(c.NUM_GOODS): # 0 to 4
                     if current_p.goods[g_id] > 0:
                         # Valid if not already in house (unless Office - hook later)
                         in_house = False
                         for h_g in gs.trading_house:
                             if h_g == g_id:
                                 in_house = True
                                 break
                         
                         if not in_house or has_office:
                             mask[c.ACTION_SELL_CORN + g_id] = 1
                             can_sell = True
            
            if not can_sell:
                mask[c.ACTION_PASS] = 1
            else:
                mask[c.ACTION_PASS] = 0 # Must sell
        
        elif gs.phase == c.PHASE_CAPTAIN:
            # Mandatory Shipping
            can_ship = False
            
            # Check Wharf
            has_wharf = False
            if not current_p.wharf_used:
                for slot in current_p.city:
                    if slot['building'] == c.BUILDING_WHARF and slot['workers'] > 0:
                        has_wharf = True
                        break

            for g_id in range(c.NUM_GOODS):
                if current_p.goods[g_id] > 0:
                    valid_ship = False
                    # Check Normal Ships
                    for s_idx, ship in enumerate(gs.ships):
                        if ship['good'] == g_id and ship['count'] < ship['capacity']:
                            valid_ship = True
                            break
                        elif ship['good'] == -1:
                             # Check other ships for this good
                             other_has = False
                             for other_s in gs.ships:
                                 if other_s['good'] == g_id:
                                     other_has = True
                                     break
                             if not other_has:
                                 valid_ship = True
                                 break
                    
                    if not valid_ship and has_wharf:
                        valid_ship = True # Can use Wharf
                        
                    if valid_ship:
                             mask[c.ACTION_SHIP_CORN + g_id] = 1
                             can_ship = True
                             # Found valid ship for this good
                             # break # Can't break, need to check other goods
            
            if not can_ship:
                mask[c.ACTION_PASS] = 1
            else:
                mask[c.ACTION_PASS] = 0 # Must ship

        elif gs.phase == c.PHASE_BUILDER:
            # Check money vs building costs
            # Check slots availability (12 slots)
            
            # Can Pass? Yes.
            mask[c.ACTION_PASS] = 1
            
            # Check if city full
            slots_filled = sum(1 for s in current_p.city if s['building'] != -1)
            if slots_filled < 12:
                # Iterate all buildings
                for b_id in range(c.NUM_BUILDINGS):
                    # Check 1: Already built?
                    already_built = False
                    for slot in current_p.city:
                        if slot['building'] == b_id:
                            already_built = True
                            break
                    if already_built:
                        continue
                        
                    # Check 2: Supply Available?
                    if gs.building_supply[b_id] <= 0:
                        continue
                        
                    # Check 3: Affordability
                    cost = c.BUILDING_INFO[b_id][0]
                    # Calc Discount
                    # Helper func? Inline for now
                    is_selector = (gs.current_role_privilege and gs.current_player_idx == gs.action_queue[0])
                    # Actually gs.current_role_privilege IS identifying the selector current turn.
                    if gs.current_role_privilege:
                         cost -= 1
                         
                    quarries = 0
                    for slot in current_p.island:
                        if slot['tile'] == c.PLANTATION_QUARRY and slot['workers'] > 0:
                            quarries += 1
                    
                    limit = c.BUILDING_INFO[b_id][3]
                    actual_discount = min(quarries, limit)
                    cost -= actual_discount
                    cost = max(0, cost)
                    
                    if current_p.doubloons >= cost:
                        mask[c.ACTION_BUILD_START + b_id] = 1

        elif gs.phase == c.PHASE_CRAFTSMAN:
             # Only Selector gets action (Bonus)
             # Mask based on produced good types
             for g_id in range(c.NUM_GOODS):
                 if current_p.last_produced_goods[g_id] > 0 and gs.supply_goods[g_id] > 0:
                     if g_id == c.CORN: mask[c.ACTION_CRAFTSMAN_BONUS_CORN] = 1
                     elif g_id == c.FRUIT: mask[c.ACTION_CRAFTSMAN_BONUS_FRUIT] = 1
                     elif g_id == c.SUGAR: mask[c.ACTION_CRAFTSMAN_BONUS_SUGAR] = 1
                     elif g_id == c.TOBACCO: mask[c.ACTION_CRAFTSMAN_BONUS_TOBACCO] = 1
                     elif g_id == c.COFFEE: mask[c.ACTION_CRAFTSMAN_BONUS_COFFEE] = 1
                     
        # Placeholder for other phases - allow PASS to prevent deadlock in tests
        elif gs.phase in [c.PHASE_TRADER, c.PHASE_CAPTAIN, c.PHASE_PROSPECTOR]:
             mask[c.ACTION_PASS] = 1
             
        return mask

    def step(self, action):
        gs = self.game_state
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # 1. Validate Action
        mask = self.get_action_mask()
        if mask[action] == 0:
             # Invalid action: return simple penalty or error? 
             # For Gym, usually undefined behavior or no-op. I'll return penalty and no state change?
             # Or just raise error for debug.
             # Let's invalid action -> large negative reward and terminate? Or just ignore.
             # I will raise Error for now to catch logic bugs in test.
             pass
             # raise ValueError(f"Invalid action {action} for phase {gs.phase} and player {gs.current_player_idx}")

        # 2. Logic Dispatch
        if gs.phase == c.PHASE_ROLE_SELECTION:
            self._step_role_selection(action)
        elif gs.phase == c.PHASE_SETTLER:
            self._step_settler(action)
        elif gs.phase == c.PHASE_MAYOR:
            self._step_mayor(action)
        elif gs.phase == c.PHASE_BUILDER:
            self._step_builder(action)
        elif gs.phase == c.PHASE_CRAFTSMAN:
            self._step_craftsman_bonus(action)
        elif gs.phase == c.PHASE_TRADER:
            self._step_trader(action)
        elif gs.phase == c.PHASE_CAPTAIN:
            self._step_captain(action)
            
        # Default placeholder logic
        elif gs.phase == c.PHASE_GAME_END:
            terminated = True
        else:
            self._advance_queue()
            
        # 3. Update Observation
        obs = self._get_obs()
        pass 
        # obs["action_mask"] = self.get_action_mask() # Removed. Handled by ActionMasker.
        
        return obs, reward, terminated, truncated, info

    def _step_role_selection(self, action):
        gs = self.game_state
        role_id = action - c.ACTION_CHOOSE_ROLE_SETTLER
        
        # Mark role taken
        gs.roles_available[role_id] = False
        gs.current_role = role_id
        
        # Give money on role to player
        doubloons = gs.roles_doubloons[role_id]
        gs.players[gs.current_player_idx].doubloons += doubloons
        gs.roles_doubloons[role_id] = 0
        
        # Setup Phase Actions
        if role_id == c.SETTLER:
            gs.phase = c.PHASE_SETTLER
            # Queue: [Selector, Other]
            selector = gs.current_player_idx
            other = (selector + 1) % c.NUM_PLAYERS
            gs.action_queue = [selector, other]
            gs.current_role_privilege = True
            
        elif role_id == c.MAYOR:
            gs.phase = c.PHASE_MAYOR
            selector = gs.current_player_idx
            other = (selector + 1) % c.NUM_PLAYERS
            gs.action_queue = [selector, other]
            gs.current_role_privilege = True
            
            # Privilege: +1 Colonist from supply
            if gs.supply_colonists > 0:
                gs.players[selector].san_juan_workers += 1
                gs.supply_colonists -= 1
            
            # Distribute Colonist Ship
            # Round robin starting from Selector
            temp_order = [selector, other]
            idx = 0
            while gs.colonist_ship > 0:
                p_idx = temp_order[idx % 2]
                gs.players[p_idx].san_juan_workers += 1
                gs.colonist_ship -= 1
                idx += 1
                
            # Initialize First Player for Placement
            # "Lift" all colonists for the first player in queue
            self._prepare_mayor_placement(gs.action_queue[0])
            
        elif role_id == c.PROSPECTOR:
             # Instant 1 doubloon for selector
             gs.players[gs.current_player_idx].doubloons += 1
             # Rulebook 139: "other players do nothing"
             self._end_role_phase()
             return
        
        elif role_id == c.TRADER:
            gs.phase = c.PHASE_TRADER
            selector = gs.current_player_idx
            other = (selector + 1) % c.NUM_PLAYERS
            gs.action_queue = [selector, other]
            gs.current_role_privilege = True
            
        elif role_id == c.CAPTAIN:
            gs.phase = c.PHASE_CAPTAIN
            selector = gs.current_player_idx
            other = (selector + 1) % c.NUM_PLAYERS
            # Captain Phase is Cyclic.
            # We start with [Selector, Other]. 
            # Logic: If queue empties, we refill it IF the round isn't done?
            # Better: Queue is dynamic. _step_captain handles refilling.
            gs.action_queue = [selector, other]
            gs.current_role_privilege = True
            # Reset consecutive passes for Captain Phase loop detection
            gs.captain_consecutive_passes = 0

        elif role_id == c.BUILDER:
            gs.phase = c.PHASE_BUILDER
            selector = gs.current_player_idx
            other = (selector + 1) % c.NUM_PLAYERS
            gs.action_queue = [selector, other]
            gs.current_role_privilege = True
            
        elif role_id == c.CRAFTSMAN:
            gs.phase = c.PHASE_CRAFTSMAN
            selector = gs.current_player_idx
            
            # Craftsman production happens IMMEDIATELY for ALL players at start of phase
            self._execute_production()
            
            produced_goods = gs.players[selector].last_produced_goods # Need to add this to PlayerState
            if any(produced_goods):
                gs.action_queue = [selector]
                gs.current_role_privilege = True 
            else:
                self._end_role_phase()
                return

        else:
            # Other roles placeholders (None left?)
            gs.phase = c.PHASE_GAME_END # Temporary lock
            gs.action_queue = [gs.current_player_idx]
            
        # Set next active player
        if gs.action_queue:
            gs.current_player_idx = gs.action_queue[0]

    def _execute_production(self):
        gs = self.game_state
        for p_idx, p in enumerate(gs.players):
             # Calculate Production
             # 1. Corn: Count Occupied Corn Plantations
             corn_count = 0
             for slot in p.island:
                 if slot['tile'] == c.PLANTATION_CORN and slot['workers'] > 0:
                     corn_count += 1
             
             # 2. Others: Min(Occupied Plantations, Occupied Factory Slots)
             # Plantations
             counts = {c.FRUIT: 0, c.SUGAR: 0, c.TOBACCO: 0, c.COFFEE: 0}
             mappings = {
                 c.PLANTATION_FRUIT: c.FRUIT,
                 c.PLANTATION_SUGAR: c.SUGAR, 
                 c.PLANTATION_TOBACCO: c.TOBACCO, 
                 c.PLANTATION_COFFEE: c.COFFEE
             }
             for slot in p.island:
                 if slot['tile'] in mappings and slot['workers'] > 0:
                     counts[mappings[slot['tile']]] += 1
             
             # Factories
             capacities = {c.FRUIT: 0, c.SUGAR: 0, c.TOBACCO: 0, c.COFFEE: 0}
             # Buildings that produce
             # Small/Large Fruit (ID 0, 2) -> Fruit
             # Small/Large Sugar (ID 1, 3) -> Sugar
             # Tobacco (ID 4) -> Tobacco
             # Coffee (ID 5) -> Coffee
             
             b_map = {
                 c.BUILDING_SMALL_FRUIT: c.FRUIT, c.BUILDING_LARGE_FRUIT: c.FRUIT,
                 c.BUILDING_SMALL_SUGAR: c.SUGAR, c.BUILDING_LARGE_SUGAR: c.SUGAR,
                 c.BUILDING_TOBACCO: c.TOBACCO,
                 c.BUILDING_COFFEE: c.COFFEE
             }
             
             for slot in p.city:
                 if slot['building'] in b_map:
                     good_type = b_map[slot['building']]
                     capacities[good_type] += slot['workers']
                     
             # Final Production (limited by Supply)
             produced = [0] * c.NUM_GOODS
             
             # Corn (No factory needed)
             actual_corn = min(corn_count, gs.supply_goods[c.CORN])
             produced[c.CORN] = actual_corn
             gs.supply_goods[c.CORN] -= actual_corn
             p.goods[c.CORN] += actual_corn
             
             for g_id in [c.FRUIT, c.SUGAR, c.TOBACCO, c.COFFEE]:
                 potential = min(counts[g_id], capacities[g_id])
                 actual = min(potential, gs.supply_goods[g_id])
                 produced[g_id] = actual
                 gs.supply_goods[g_id] -= actual
                 p.goods[g_id] += actual
                 
             p.last_produced_goods = produced 
             
             # Factory Bonus (Line 261: Factory Building)
             kinds_produced = sum(1 for x in produced if x > 0)
             if kinds_produced >= 2:
                 # Check if player has Occupied Factory
                 has_factory = False
                 for slot in p.city:
                     if slot['building'] == c.BUILDING_FACTORY and slot['workers'] > 0:
                         has_factory = True
                         break
                 if has_factory:
                     bonus = 0
                     if kinds_produced == 2: bonus = 1
                     elif kinds_produced == 3: bonus = 2
                     elif kinds_produced == 4: bonus = 3
                     elif kinds_produced == 5: bonus = 5
                     p.doubloons += bonus

    def _prepare_mayor_placement(self, p_idx):
        # Move all colonists from Board to San Juan (Pool)
        p = self.game_state.players[p_idx]
        count = 0
        for slot in p.island:
            count += slot['workers']
            slot['workers'] = 0
        for slot in p.city:
            count += slot['workers']
            slot['workers'] = 0
        
        p.san_juan_workers += count

    def _step_settler(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        tile_to_take = -1
        is_quarry = False
        
        if action == c.ACTION_SETTLER_TAKE_QUARRY:
            if gs.supply_quarries > 0:
                is_quarry = True
                gs.supply_quarries -= 1
                
        elif c.ACTION_SETTLER_TAKE_PLANTATION_0 <= action <= c.ACTION_SETTLER_TAKE_PLANTATION_2:
            idx = action - c.ACTION_SETTLER_TAKE_PLANTATION_0
            if idx < len(gs.market_plantations):
                tile_to_take = gs.market_plantations.pop(idx)
        
        elif action == c.ACTION_PASS:
            pass
            
        # Place on board
        if is_quarry:
            target_slot = -1
            # Find empty island slot
            for i, slot in enumerate(current_p.island):
                if slot['tile'] == -1:
                    slot['tile'] = c.PLANTATION_QUARRY
                    target_slot = i
                    break
            
            if target_slot != -1:
                # Check Hospice
                has_hospice = False
                for slot in current_p.city:
                    if slot['building'] == c.BUILDING_HOSPICE and slot['workers'] > 0:
                        has_hospice = True
                        break
                
                if has_hospice:
                    if gs.supply_colonists > 0:
                        current_p.island[target_slot]['workers'] = 1
                        gs.supply_colonists -= 1
                    elif gs.colonist_ship > 0:
                        current_p.island[target_slot]['workers'] = 1
                        gs.colonist_ship -= 1

        elif tile_to_take != -1:
             target_slot = -1
             for i, slot in enumerate(current_p.island):
                if slot['tile'] == -1:
                    slot['tile'] = tile_to_take
                    target_slot = i
                    break
             
             if target_slot != -1:
                # Check Hospice
                has_hospice = False
                for slot in current_p.city:
                    if slot['building'] == c.BUILDING_HOSPICE and slot['workers'] > 0:
                        has_hospice = True
                        break
                
                if has_hospice:
                    if gs.supply_colonists > 0:
                        current_p.island[target_slot]['workers'] = 1
                        gs.supply_colonists -= 1
                    elif gs.colonist_ship > 0:
                        current_p.island[target_slot]['workers'] = 1
                        gs.colonist_ship -= 1
        
        # Hacienda Ability: If occupied, draw 1 extra random tile from deck
        has_hacienda = False
        for slot in current_p.city:
            if slot['building'] == c.BUILDING_HACIENDA and slot['workers'] > 0:
                has_hacienda = True
                break
        
        if has_hacienda:
            # Draw from deck
            has_hospice = False 
            for slot in current_p.city:
                if slot['building'] == c.BUILDING_HOSPICE and slot['workers'] > 0:
                    has_hospice = True
                    break

            extra_tile = -1
            if gs.plantation_deck:
                extra_tile = gs.plantation_deck.pop()
            elif gs.discarded_plantations:
                 random.shuffle(gs.discarded_plantations)
                 gs.plantation_deck.extend(gs.discarded_plantations)
                 gs.discarded_plantations = []
                 if gs.plantation_deck:
                     extra_tile = gs.plantation_deck.pop()
            
            if extra_tile != -1:
                # Place on island
                for i, slot in enumerate(current_p.island):
                    if slot['tile'] == -1:
                        slot['tile'] = extra_tile
                        if has_hospice:
                             if gs.supply_colonists > 0:
                                slot['workers'] = 1
                                gs.supply_colonists -= 1
                             elif gs.colonist_ship > 0:
                                slot['workers'] = 1
                                gs.colonist_ship -= 1
                        break
                    
        self._advance_queue()

    def _step_mayor(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        if action == c.ACTION_PASS:
            # Done placing
            self._advance_queue()
            if gs.phase == c.PHASE_MAYOR: # If still in phase, prep next player
                self._prepare_mayor_placement(gs.current_player_idx)
            return

        # Place Colonist Logic
        target_type = None # 'island' or 'city'
        target_idx = -1
        
        if c.ACTION_MAYOR_PLACE_PLANTATION_0 <= action <= c.ACTION_MAYOR_PLACE_PLANTATION_11:
            target_type = 'island'
            target_idx = action - c.ACTION_MAYOR_PLACE_PLANTATION_0
        elif c.ACTION_MAYOR_PLACE_BUILDING_0 <= action <= c.ACTION_MAYOR_PLACE_BUILDING_11:
            target_type = 'city'
            target_idx = action - c.ACTION_MAYOR_PLACE_BUILDING_0
            
        if target_type == 'island':
            # Check validity
            if current_p.san_juan_workers > 0 and 0 <= target_idx < 12:
                slot = current_p.island[target_idx]
                # Can only place if tile exists and is empty (or rule says max 1 worker?)
                # Rulebook Line 67: "Each circle can hold exactly 1 colonist"
                # So max 1.
                if slot['tile'] != -1 and slot['workers'] == 0:
                    slot['workers'] = 1
                    current_p.san_juan_workers -= 1
                    
        elif target_type == 'city':
             if current_p.san_juan_workers > 0 and 0 <= target_idx < 12:
                slot = current_p.city[target_idx]
                if slot['building'] != -1:
                    # Check capacity
                    b_id = slot['building']
                    capacity = c.BUILDING_INFO[b_id][2]
                    if slot['workers'] < capacity:
                        slot['workers'] += 1
                        current_p.san_juan_workers -= 1
                        
        # Stay in Mayor Phase for this player until they Pass or run out?
        # Typically one action per step. So we return.
        # Player must continue until they decide to Pass.
        pass

    def _advance_queue(self):
        gs = self.game_state
        # Remove current actor
        if gs.action_queue:
            gs.action_queue.pop(0)
            
        if gs.action_queue:
            # Next player in queue
            gs.current_player_idx = gs.action_queue[0]
            gs.current_role_privilege = False # Privilege only for first actor
        else:
            # End of Role Phase
            self._end_role_phase()
            
    def _end_role_phase(self):
        gs = self.game_state
        
        if gs.phase == c.PHASE_SETTLER:
            # Refill plantatons
            gs.discarded_plantations.extend(gs.market_plantations)
            gs.market_plantations = []
            for _ in range(3):
                if gs.plantation_deck:
                    gs.market_plantations.append(gs.plantation_deck.pop())
                elif gs.discarded_plantations:
                    random.shuffle(gs.discarded_plantations)
                    gs.plantation_deck.extend(gs.discarded_plantations)
                    gs.discarded_plantations = []
                    if gs.plantation_deck:
                        gs.market_plantations.append(gs.plantation_deck.pop())

        elif gs.phase == c.PHASE_MAYOR:
            # Refill Colonist Ship
            # Count empty slots on all players buildings
            total_empty = 0
            for p in gs.players:
                for slot in p.city:
                    if slot['building'] != -1:
                        cap = c.BUILDING_INFO[slot['building']][2]
                        total_empty += (cap - slot['workers'])
            
            fill_amount = max(total_empty, c.NUM_PLAYERS) # Min 2 for 2 players
            
            if gs.supply_colonists < fill_amount:
                # Not enough colonists
                gs.colonist_ship = gs.supply_colonists
                gs.supply_colonists = 0
                # Game End Trigger 1 (Rulebook 53/144)
                # "When... cannot be refilled entirely... game ends at END OF ROUND"
                # Need to mark game end flag
                gs.game_end_triggered = True
            else:
                gs.colonist_ship = fill_amount
                gs.supply_colonists -= fill_amount

        gs.roles_taken_count += 1
        
        if gs.roles_taken_count >= 6:
            self._end_round()
        else:
            gs.phase = c.PHASE_ROLE_SELECTION
            gs.current_player_idx = (gs.governor_idx + gs.roles_taken_count) % c.NUM_PLAYERS
            
    def _end_round(self):
        gs = self.game_state
        
        # Check Game End
        if getattr(gs, 'game_end_triggered', False):
            gs.phase = c.PHASE_GAME_END
            return
            
        # 1. 1 Doubloon on unchosen roles
        for r_id in range(c.NUM_ROLES):
            if gs.roles_available[r_id]:
                gs.roles_doubloons[r_id] += 1
                
        # 2. Reset Roles
        gs.roles_available = [True] * c.NUM_ROLES
        
        # 3. Change Governor
        gs.governor_idx = (gs.governor_idx + 1) % c.NUM_PLAYERS
        
        # 4. Reset counters
        gs.roles_taken_count = 0
        gs.phase = c.PHASE_ROLE_SELECTION
        gs.current_player_idx = gs.governor_idx

 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        
        self.game_state = GameState()
        
        # Setup Plantation Deck
        # Rulebook: 
        # Coffee 5, Tobacco 6, Corn 7, Sugar 8, Fruit 9
        # "농장 타일 35개를 잘 섞고"
        deck = []
        
        counts = c.PLANTATION_COUNTS
        for p_id, count in counts.items():
            deck.extend([p_id] * count)
        
        random.shuffle(deck)
        
        # Players setup
        p1 = PlayerState()
        p2 = PlayerState()
        self.game_state.players = [p1, p2]
        
        # Initial Plantations
        # Governor (P1) gets Fruit (Indigo)
        # P2 gets Corn
        # But wait, deck has 35 total usually.
        # Rulebook: "각 플레이어는... 농장 타일 1개를 가져감"
        # "첫 라운드의 시작 플레이어는 과일 타일... 다른 플레이어는 옥수수 타일"
        # These are taken from the supply *before* creating the deck? Or from the deck?
        # Rulebook phrasing: "농장 타일 35개를 잘 섞고 공급처를 만든 후...".
        # Standard PR: Start plantations are separate from the deck.
        # Rulebook phrasing: "농장 타일 35개를 잘 섞고... (Meanwhile p1 takes fruit...)"
        # Implies STARTING plantations are EXTRA or taken OUT.
        # Given "35" is standard count (8+5+9+7+6 = 35), and players take 1 each.
        # Usually setup is: Give players starting tiles. THEN shuffle remainder.
        # I will remove 1 Corn and 1 Fruit from the "Counts" to simulated them being taken, or just assume they are separate. 
        # Actually standard rules: The start corn/indigo are PART of the total component count.
        # So I will reduce the deck by what players took.
        
        start_p1_tile = c.PLANTATION_FRUIT
        start_p2_tile = c.PLANTATION_CORN
        
        # Remove from deck logic: Handled by just creating deck with full counts and removing specific ones? 
        # Easier: Create full deck, find and remove.
        # Or: Decrement counts.
        
        # Let's decrement counts for deck creation safely.
        deck_counts = c.PLANTATION_COUNTS.copy()
        deck_counts[start_p1_tile] -= 1
        deck_counts[start_p2_tile] -= 1
        
        # Re-build deck
        self.game_state.plantation_deck = []
        for p_id, count in deck_counts.items():
            self.game_state.plantation_deck.extend([p_id] * count)
        
        random.shuffle(self.game_state.plantation_deck)
        
        # Give to players
        p1.island[0] = {'tile': start_p1_tile, 'workers': 0}
        p2.island[0] = {'tile': start_p2_tile, 'workers': 0}
        
        # Reveal 3 market plantations (Rulebook Line 20: "타일 3개를 공개함")
        for _ in range(3):
            if self.game_state.plantation_deck:
                self.game_state.market_plantations.append(self.game_state.plantation_deck.pop())
                
        return self._get_obs(), {}

    def _step_builder(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        if action == c.ACTION_PASS:
            pass
        elif c.ACTION_BUILD_START <= action < c.ACTION_BUILD_START + c.NUM_BUILDINGS:
            b_id = action - c.ACTION_BUILD_START
            
            # Sanity Calc Cost
            cost = c.BUILDING_INFO[b_id][0]
            # Builder Discount
            if gs.current_role_privilege and gs.current_player_idx == gs.action_queue[0]: # Wait, privilege is defined in role setup
                if gs.current_role_privilege:
                    cost -= 1
            
            # Quarry Discount
            quarry_discount = 0
            quarries = 0
            for slot in current_p.island:
                if slot['tile'] == c.PLANTATION_QUARRY and slot['workers'] > 0:
                    quarries += 1
            
            limit = c.BUILDING_INFO[b_id][3]
            actual_discount = min(quarries, limit)
            cost -= actual_discount
            cost = max(0, cost)
            
            # Pay
            if current_p.doubloons >= cost:
                current_p.doubloons -= cost
                # Place
                target_slot_idx = -1
                for i, slot in enumerate(current_p.city):
                    if slot['building'] == -1:
                        slot['building'] = b_id
                        target_slot_idx = i
                        break
                # Remove from supply
                if gs.building_supply[b_id] > 0:
                    gs.building_supply[b_id] -= 1
                
                # University Ability: Get 1 colonist
                if target_slot_idx != -1:
                     has_university = False
                     for slot in current_p.city:
                         if slot['building'] == c.BUILDING_UNIVERSITY and slot['workers'] > 0:
                             has_university = True
                             break
                     if has_university and gs.supply_colonists > 0:
                         current_p.city[target_slot_idx]['workers'] = 1
                         gs.supply_colonists -= 1
                
                # Check for Game End (12 buildings)
                # Count filled slots
                filled = sum(1 for s in current_p.city if s['building'] != -1)
                if filled >= 12:
                    gs.game_end_triggered = True
        
        self._advance_queue()

    def _step_craftsman_bonus(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        # Only selector gets here
        good_id = -1
        if action == c.ACTION_CRAFTSMAN_BONUS_CORN: good_id = c.CORN
        elif action == c.ACTION_CRAFTSMAN_BONUS_FRUIT: good_id = c.FRUIT
        elif action == c.ACTION_CRAFTSMAN_BONUS_SUGAR: good_id = c.SUGAR
        elif action == c.ACTION_CRAFTSMAN_BONUS_TOBACCO: good_id = c.TOBACCO
        elif action == c.ACTION_CRAFTSMAN_BONUS_COFFEE: good_id = c.COFFEE
        
        if good_id != -1:
             # Basic validity check (produced? supply?)
             # Logic is trusted to be masked correctly or simple check here
             if current_p.last_produced_goods[good_id] > 0 and gs.supply_goods[good_id] > 0:
                 current_p.goods[good_id] += 1
                 gs.supply_goods[good_id] -= 1
                 
        self._end_role_phase()

    def _step_trader(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        if action == c.ACTION_PASS:
            pass
        elif c.ACTION_SELL_CORN <= action <= c.ACTION_SELL_COFFEE:
            # Map action to Good ID
            good_map = {
                c.ACTION_SELL_CORN: c.CORN,
                c.ACTION_SELL_FRUIT: c.FRUIT,
                c.ACTION_SELL_SUGAR: c.SUGAR,
                c.ACTION_SELL_TOBACCO: c.TOBACCO,
                c.ACTION_SELL_COFFEE: c.COFFEE
            }
            good_id = good_map[action]
            
            # Logic Check (Assume Mask is Correct, but verify basics)
            if current_p.goods[good_id] > 0:
                # Sell
                current_p.goods[good_id] -= 1
                gs.supply_goods[good_id] += 1
                
                # Place in House
                for i in range(4):
                    if gs.trading_house[i] == -1:
                        gs.trading_house[i] = good_id
                        break
                        
                # Money
                prices = [0, 1, 2, 3, 4] # Corn=0, Fruit=1...
                doubloons = prices[good_id]
                
                # Office Bonus (+1 if occupied Office) (TODO later: Office Logic Hook)
                # Small Market (+1), Large Market (+2) - Wait, these are buildings.
                # Rulebook: "Small Market: +1 dbl on sale", "Large Market: +2 dbl".
                # Check buildings
                market_bonus = 0
                for slot in current_p.city:
                    if slot['workers'] > 0:
                        if slot['building'] == c.BUILDING_SMALL_MARKET: market_bonus += 1
                        elif slot['building'] == c.BUILDING_LARGE_MARKET: market_bonus += 2
                        elif slot['building'] == c.BUILDING_OFFICE: market_bonus += 1 # Office bonus
                
                doubloons += market_bonus
                
                # Privilege: Selector gets +1
                if gs.current_role_privilege and gs.current_player_idx == gs.action_queue[0]: # Wait, safer: if queue[0] == current
                     # Wait, action_queue might have changed? No in Trader it's linear.
                     # But current_role_privilege is reset after first player.
                     if gs.current_role_privilege:
                         doubloons += 1
                
                current_p.doubloons += doubloons

        self._advance_queue()

    def _step_captain(self, action):
        gs = self.game_state
        current_p = gs.players[gs.current_player_idx]
        
        did_ship = False
        
        if action == c.ACTION_PASS:
            gs.captain_consecutive_passes += 1
        elif c.ACTION_SHIP_CORN <= action <= c.ACTION_SHIP_COFFEE:
             # Ship Logic
             good_map = {
                c.ACTION_SHIP_CORN: c.CORN,
                c.ACTION_SHIP_FRUIT: c.FRUIT,
                c.ACTION_SHIP_SUGAR: c.SUGAR,
                c.ACTION_SHIP_TOBACCO: c.TOBACCO,
                c.ACTION_SHIP_COFFEE: c.COFFEE
             }
             good_id = good_map[action]
             
             # Calculate Normal Shipping Option
             best_ship_idx = -1
             normal_ship_amount = 0
             
             for s_idx, ship in enumerate(gs.ships):
                 if ship['good'] == good_id and ship['count'] < ship['capacity']:
                     amount = min(current_p.goods[good_id], ship['capacity'] - ship['count'])
                     if amount > 0:
                         best_ship_idx = s_idx
                         normal_ship_amount = amount
                         break # Take first valid
                 elif ship['good'] == -1:
                     # Check others
                     other_has = False
                     for other_s in gs.ships:
                         if other_s['good'] == good_id:
                             other_has = True
                             break
                     if not other_has:
                         amount = min(current_p.goods[good_id], ship['capacity'])
                         if amount > 0:
                             best_ship_idx = s_idx
                             normal_ship_amount = amount
                             break # Take first valid
             
             # Calculate Wharf Shipping Option
             wharf_amount = 0
             has_wharf = False
             if not current_p.wharf_used:
                  for slot in current_p.city:
                      if slot['building'] == c.BUILDING_WHARF and slot['workers'] > 0:
                          has_wharf = True
                          break
             
             if has_wharf:
                 wharf_amount = current_p.goods[good_id]
            
             # Decision: Use Wharf if Better than Normal
             use_wharf = False
             if has_wharf and wharf_amount > normal_ship_amount:
                 use_wharf = True
             elif best_ship_idx == -1 and has_wharf and wharf_amount > 0:
                 use_wharf = True
             elif best_ship_idx != -1:
                 # Default to Normal
                 use_wharf = False
                 
             # Execute
             if use_wharf:
                 ship_amount = wharf_amount
                 current_p.goods[good_id] -= ship_amount
                 gs.supply_goods[good_id] += ship_amount
                 current_p.wharf_used = True
                 # Wharf allows shipping ALL. No ship capacity limit.
                 # No ship modified.
             elif best_ship_idx != -1:
                 ship = gs.ships[best_ship_idx]
                 ship_amount = normal_ship_amount
                 current_p.goods[good_id] -= ship_amount
                 gs.supply_goods[good_id] += ship_amount
                 if ship['good'] == -1:
                     ship['good'] = good_id
                 ship['count'] += ship_amount
             else:
                 # Should not happen if Mask is correct
                 ship_amount = 0
             
             if ship_amount > 0:
                 # VP
                 points = ship_amount
                 
                 # Harbor Bonus (+1 VP extra per shipment)
                 has_harbor = False
                 for slot in current_p.city:
                     if slot['building'] == c.BUILDING_HARBOR and slot['workers'] > 0:
                         has_harbor = True
                         break
                 if has_harbor:
                     points += 1
                 
                 current_p.vp_chips += points
                 gs.supply_vp -= points
                 if gs.supply_vp <= 0:
                     gs.game_end_triggered = True
                 
                 # Captain Privilege
                 if gs.current_role_privilege:
                     current_p.vp_chips += 1
                     gs.supply_vp -= 1
                     if gs.supply_vp <= 0:
                         gs.game_end_triggered = True
                     # Privilege used? Logic handles `gs.current_role_privilege` via `_advance_queue`.
                     # But in Captain, queue is cyclic.
                     # We must ensure Privilege is only ONCE.
                     # `_advance_queue` sets `current_role_privilege` to False. 
                     # Wait, `_advance_queue` implementation: 
                     # "gs.current_role_privilege = False" (Line 285 in previous view).
                     # So subsequent actions in cyclic queue will NOT have privilege. Correct.
                 
                 did_ship = True
                 gs.captain_consecutive_passes = 0
        
        if did_ship:
             pass
        
        if gs.captain_consecutive_passes < c.NUM_PLAYERS:
            gs.action_queue.append(gs.current_player_idx)
        else:
             self._end_captain_phase()
             return

        self._advance_queue()

    def _end_captain_phase(self):
        gs = self.game_state
        
        # 1. Full Ships Empty
        for ship in gs.ships:
            if ship['count'] == ship['capacity']:
                ship['good'] = -1
                ship['count'] = 0
                
        # 2. Rotting (Keep 1 + Warehouse Protection)
        for p in gs.players:
            # Check Warehouses
            protect_kinds = 0
            for slot in p.city:
                if slot['workers'] > 0:
                    if slot['building'] == c.BUILDING_SMALL_WAREHOUSE:
                        protect_kinds += 1
                    elif slot['building'] == c.BUILDING_LARGE_WAREHOUSE:
                        protect_kinds += 2
                        
            # Strategy: Keep types with most quantity fully (up to protect_kinds).
            # Then keep 1 of another type.
            # Discard rest.
            
            # Identify held goods
            held_types = []
            for g_id in range(c.NUM_GOODS):
                if p.goods[g_id] > 0:
                    held_types.append((g_id, p.goods[g_id]))
            
            # Sort by Quantity Descending (Keep most valuable volume)
            held_types.sort(key=lambda x: x[1], reverse=True)
            
            kinds_to_keep_fully = protect_kinds
            
            processed_types = 0
            has_kept_one_unit = False
            
            for g_id, qty in held_types:
                if processed_types < kinds_to_keep_fully:
                    # Keep all
                    processed_types += 1
                else:
                    # Keep 1 if haven't yet
                    if not has_kept_one_unit:
                        if qty > 1:
                            # Discard qty-1
                            p.goods[g_id] = 1
                            gs.supply_goods[g_id] += (qty - 1)
                        has_kept_one_unit = True
                    else:
                        # Discard all
                        p.goods[g_id] = 0
                        gs.supply_goods[g_id] += qty
            
            # Reset Wharf usage
            p.wharf_used = False
            
        self._end_role_phase()

    def _calculate_score(self):
        gs = self.game_state
        scores = {}
        tie_breakers = {}
        for p_idx, p in enumerate(gs.players):
            score = 0
            # 1. VP Chips
            score += p.vp_chips
            
            # 2. Building VP
            occupied_count = 0 
            # Need to count occupied for bonus buildings?
            # Rulebook: "Large buildings... extra VP if occupied."
            # Base VP is always counted? Rulebook: "VP value... on the building."
            # "Occupied" is only for Special Functions?
            # Rulebook: "At game end... VP for his buildings..."
            # Base VP applies whether occupied or not.
            
            for slot in p.city:
                if slot['building'] != -1:
                    b_id = slot['building']
                    score += c.BUILDING_INFO[b_id][1]
                    
                    # 3. Bonus VP (Large Buildings) - ONLY IF OCCUPIED
                    if c.BUILDING_INFO[b_id][3] == 4 and slot['workers'] > 0: # Large buildings have quarry limit 4? No.
                        # Check ID ranges or specific IDs
                        # Guild Hall, Residence, Fortress, Customs, City Hall
                        if b_id == c.BUILDING_GUILD_HALL:
                            # 1 VP for Small Production (occupied or not? "per production building")
                            # 2 VP for Large Production
                            total_prod_vp = 0
                            for ps in p.city:
                                if ps['building'] != -1:
                                    pid = ps['building']
                                    # Small Prod: Small Indigo(Unused?), Small Sugar(1), Small Fruit(0)
                                    # Large Prod: Factory(Unused?), Large Sugar(3), Large Fruit(2), Tobacco, Coffee
                                    # Wait, Factory is Production? No, Industrial.
                                    # Production Buildings Rulebook: "Indentured... Small/Large Indigo, Sugar..."
                                    # Checking IDs:
                                    # Small Fruit(0), Small Sugar(1).
                                    # Large Fruit(2), Large Sugar(3).
                                    # Tobacco(4), Coffee(5).
                                    # Are there others?
                                    if pid in [c.BUILDING_SMALL_FRUIT, c.BUILDING_SMALL_SUGAR]:
                                        total_prod_vp += 1
                                    elif pid in [c.BUILDING_LARGE_FRUIT, c.BUILDING_LARGE_SUGAR, c.BUILDING_TOBACCO, c.BUILDING_COFFEE]:
                                        total_prod_vp += 2
                            score += total_prod_vp
                            
                        elif b_id == c.BUILDING_RESIDENCE:
                            # VP for Plantations (Occupied or not? "filled island spaces")
                            # <10: 4 VP, 10: 5, 11: 6, 12: 7
                            filled = sum(1 for s in p.island if s['tile'] != -1)
                            if filled <= 9: score += 4
                            elif filled == 10: score += 5
                            elif filled == 11: score += 6
                            elif filled == 12: score += 7
                            
                        elif b_id == c.BUILDING_FORTRESS:
                            # 1 VP for every 3 workers
                            total_workers = sum(s['workers'] for s in p.island) + sum(s['workers'] for s in p.city) + p.san_juan_workers
                            score += (total_workers // 3)
                            
                        elif b_id == c.BUILDING_CUSTOMS_HOUSE:
                            # 1 VP for every 4 VP chips
                            score += (p.vp_chips // 4)
                            
                        elif b_id == c.BUILDING_CITY_HALL:
                            # 1 VP for each violet building (including City Hall itself?)
                            # Rulebook: "for each violet building (large/small)..."
                            violet_count = 0
                            for ps in p.city:
                                if ps['building'] != -1:
                                    # Production checks
                                    if ps['building'] not in [c.BUILDING_SMALL_FRUIT, c.BUILDING_SMALL_SUGAR, c.BUILDING_LARGE_FRUIT, c.BUILDING_LARGE_SUGAR, c.BUILDING_TOBACCO, c.BUILDING_COFFEE]:
                                        violet_count += 1
                            score += violet_count

            scores[p_idx] = score
            
            # Tie Breaker: Doubloons + Goods Count
            goods_count = sum(p.goods)
            tie_breakers[p_idx] = p.doubloons + goods_count
            
        return scores, tie_breakers

    def _get_obs(self):
        gs = self.game_state
        
        # Global Vector
        global_vec = np.zeros(self.global_space_dim, dtype=np.int32)
        global_vec[0] = gs.supply_colonists
        global_vec[1] = gs.supply_vp
        global_vec[2] = gs.supply_quarries
        global_vec[3:8] = gs.supply_goods
        
        for i in range(c.NUM_ROLES):
            global_vec[8 + i] = 1 if gs.roles_available[i] else 0
            global_vec[15 + i] = gs.roles_doubloons[i]
            
        global_vec[22:26] = gs.trading_house
        
        global_vec[26] = gs.ships[0]['good']
        global_vec[27] = gs.ships[0]['count']
        global_vec[28] = gs.ships[0]['capacity']
        
        global_vec[29] = gs.ships[1]['good']
        global_vec[30] = gs.ships[1]['count']
        global_vec[31] = gs.ships[1]['capacity']
        
        global_vec[32] = gs.governor_idx
        global_vec[33] = gs.current_player_idx
        global_vec[34] = gs.colonist_ship
        
        # Players Vector
        players_vec = np.zeros((c.NUM_PLAYERS, self.player_space_dim), dtype=np.int32)
        
        for p_idx, p in enumerate(gs.players):
            players_vec[p_idx, 0] = p.doubloons
            players_vec[p_idx, 1] = p.vp_chips
            players_vec[p_idx, 2:7] = p.goods
            
            # Island
            for idx, slot in enumerate(p.island):
                base_idx = 7 + (idx * 2)
                players_vec[p_idx, base_idx] = slot['tile']
                players_vec[p_idx, base_idx+1] = slot['workers']
                
            # City
            for idx, slot in enumerate(p.city):
                base_idx = 31 + (idx * 2)
                players_vec[p_idx, base_idx] = slot['building']
                players_vec[p_idx, base_idx+1] = slot['workers']
                
            players_vec[p_idx, 55] = p.san_juan_workers
            
        # Market Vector
        market_vec = np.zeros(3, dtype=np.int32)
        # Pad with -1 if fewer than 3
        for i in range(3):
            if i < len(gs.market_plantations):
                market_vec[i] = gs.market_plantations[i]
            else:
                market_vec[i] = -1
                
        return {
            "global": global_vec,
            "players": players_vec,
            "market_plantations": market_vec
        }
