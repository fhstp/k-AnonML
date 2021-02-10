# -*- coding: utf-8 -*-

class AnonCheck:

    def __init__(self, raw_data, max_gen, gen_strat, allowed_suppressed, k):
        """Creates the initial generalization Values

        Parameters:
            raw_data: list
                The data read from a csv file (data[col][row])
            max_gen: list
                Array of max level of each quasi-identifier (in order)
            gen_strat: list
                Array of functions to generalize each quasi-identifier (in order)
            allowed_suppressed: int
                Number of rows that are allowed to be suppressed
            k: int
                Number that defines what k-Anonymity is desired
        """
        self.gtime = 0

        self.k = k
        self.allowed_suppressed = allowed_suppressed

        self.raw_rows_count = len(raw_data[0])
        self.raw_cols_count = len(raw_data)

        self.hier_array = []
        self.data_array = []
        dict_array = []

        self.buffer = [[0 for _ in range(self.raw_cols_count)] for _ in range(self.raw_rows_count)]
        self.prev_gen_to_apply = None
        self.eq_classes_dict = None
        self.is_transformed = True

        key_rows = range(self.raw_rows_count)
        vals = [1] * self.raw_rows_count
        self.dummy_raw_eq_classes = list(zip(vals, key_rows))

        for col in range(self.raw_cols_count):
            # New list entries for this column
            self.hier_array.append([])
            self.data_array.append([])

            quasi_identifier = raw_data[col]
            quasi_identifier_set = set(quasi_identifier)

            # Create dictionary of quasi_identifier with index as key
            dict_array.append({k: v for k, v in enumerate(quasi_identifier_set)})

            # Switch key and value
            dict_reverse = dict((v, k) for k, v in dict_array[col].items())
            # Create data_array with numerical references to original data in dict_array
            for v in quasi_identifier:
                self.data_array[col].append(dict_reverse[v])

            self.test = []
            for r in range(len(self.data_array[0])):
                self.test.append([])
                for c in range(len(self.data_array)):
                    self.test[-1].append(self.data_array[c][r])

            # Fill first generalization level with the numerical reference values
            self.hier_array[col].append(list(dict_array[-1].keys()))
            # Iterate generalization levels
            for level in range(0, max_gen[col]):
                idexes = apply_generalization(quasi_identifier_set, gen_strat[col], level, dict_array[-1])
                # Append numerical reference values for this level
                self.hier_array[col].append(idexes)

    def calculate_kanon(self, node):
        gen_to_apply = node.attributes
        is_rollup_allowed = False

        raw_eq_classes = self.dummy_raw_eq_classes

        if self.eq_classes_dict is not None:
            prev_level = sum(self.prev_gen_to_apply)
            level = sum(gen_to_apply)
            if level > prev_level:
                is_rollup_allowed = True
                for num in range(len(gen_to_apply)):
                    if gen_to_apply[num] < self.prev_gen_to_apply[num]:
                        is_rollup_allowed = False
                        break

                if is_rollup_allowed:
                    self.is_transformed = False
                    # Roll-up
                    raw_eq_classes = self.eq_classes_dict.values()

        # Projection
        cols_to_iterate = range(self.raw_cols_count)
        if self.is_transformed or is_rollup_allowed:
            if self.prev_gen_to_apply is not None:
                cols_to_iterate = [i for i, value in enumerate(gen_to_apply) if value != self.prev_gen_to_apply[i]]

        if not is_rollup_allowed:
            self.is_transformed = True

        self.eq_classes_dict = {}
        update = self.eq_classes_dict.update

        for val, key_row in raw_eq_classes:
            tmp = self.buffer[key_row]
            rawr = self.test[key_row]
            for col in cols_to_iterate:
                tmp[col] = self.hier_array[col][gen_to_apply[col]][rawr[col]]

            tup = tuple(tmp)

            try:
                self.eq_classes_dict[tup][0] += val
            except KeyError:
                update({tup: [val, key_row]})

        self.prev_gen_to_apply = gen_to_apply.copy()

        suppressed_count = 0

        eq_classes = sorted(list(zip(*self.eq_classes_dict.values()))[0])
        eqsum = 0
        amount = 0
        for v in eq_classes:

            if v < self.k:
                suppressed_count += v
                if not suppressed_count > self.allowed_suppressed:
                    node.DM_penalty += v * self.raw_rows_count
            else:
                eqsum += v
                amount += 1
                node.DM_penalty += v*v
                node.DMs_penalty += v*v
        if amount == 0:
            node.eqclasses = 0
        else:
            node.eqclasses = eqsum/amount
        suppressed_count = 0
        for v in eq_classes:
            if v < self.k:
                if self.allowed_suppressed == 0:
                    return False
                suppressed_count += v
                if suppressed_count > self.allowed_suppressed:
                    return False
            else:
                return True
        return True


def apply_generalization(data, strat, level, dictionary):
    tmp_array = []
    gen_index = []
    count = len(dictionary)-1

    for v in data:
        if isinstance(strat, list):
            args = []
            for arg_len in range(1, len(strat)):
                args.append(strat[arg_len])
            vg = strat[0](v, level, *tuple(args))
        else:
            vg = strat(v, level)

        if isinstance(vg, list):
            vg = vg[0]
        if vg not in tmp_array:
            count += 1
            dictionary.update({count: vg})
            tmp_array.append(vg)
            gen_index.append(count)

        else:
            for key, val in dictionary.items():
                if vg == val:
                    gen_index.append(key)
                    break

    return gen_index
