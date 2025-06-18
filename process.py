from datetime import datetime
import pandas as pd 

class Processor:
    
    """
    A class for processing YOLO output and counting sales events.

    This class maintains tracking information for detected objects,
    counts the number of sales, and records timestamps of sales events.

    Attributes:
        tracked (dict): Stores tracking information for detected objects.
        sales (int): The total number of sales counted.
        times (list): List of timestamps when sales occurred.
    """


    def __init__(self):
        self.tracked = {}
        self.database = {
                         "Data": [],
                         "Time": [],
                         "Count": 0
                         }
        self.pizza2box = {}
        
    def get_new_sales_df(self):
        """
        Convert the latest sale(s) in self.database into
        a one‑row DataFrame, then reset that part of the database.
        """
        if self.database["Count"] == 0:
            return None
        df = pd.DataFrame([{
                            "Date":     self.database["Date"][-1],
                            "Time":     self.database["Time"][-1],
                            "Count":    self.database["Count"]
                          }])
        return df
                
    def _is_inside_box(self, openbox_coor: tuple, pizza_coor: tuple):
        """
        Check if the pizza bounding box is completely inside the open box bounding box.
        
                arguments:
                    openbox_coor: tuple (open box's xmin, ymin, xmax, ymax)
                    pizza_coor: tuple (Pizza's xmin, ymin, xmax, ymax)
        """
        
        bx_min, by_min, bx_max, by_max = openbox_coor 
        px_min, py_min, px_max, py_max = pizza_coor
        
        # The logic to check if the pizza is inside the open box
        if bx_min < px_min and by_min < py_min and bx_max > px_max and by_max > py_max:
            return True
        else:
            return False

    def _get_config(self, results: list):
        """Extracts bounding box coordinates, class IDs, and object IDs for pizzas and open boxes from YOLO results."""

        # Get classes, boxex and tracked id, then convert them to list
        clss = results[0].boxes.cls.int().tolist()
        boxes = results[0].boxes.xyxy.tolist()
        ids = results[0].boxes.id



        # Leveraging list comprehension we store bounding box for pizza and openboxes separately
        pizzas = [c for i, c in enumerate(boxes) if results[0].names[clss[i]] == "pizza"]
        openboxes = [c for i, c in enumerate(boxes) if results[0].names[clss[i]] == "box"]

        # Do the same with pizza's track ids and open box's track id
        pizza_ids = [ids[i] if ids is not None else None for i, c in enumerate(clss) if results[0].names[c] == "pizza"]
        openbox_ids = [ids[i] if ids is not None else None for i, c in enumerate(clss) if results[0].names[c] == "box"]

        return pizzas, openboxes, pizza_ids, openbox_ids    
        
    def _process_bbox(self, pizzas, openboxes, pizza_ids, openbox_ids):
        """
        Processes bounding boxes to determine if a pizza is inside an open box.
        Updates tracking information and records the time of each new sale event.
        
            Args:
                pizzas (list): List of pizza bounding box coordinates.
                openboxes (list): List of open box bounding box coordinates.
                pizza_ids (list): List of pizza object IDs.
                openbox_ids (list): List of open box object IDs.
        """
        date = datetime.now().strftime("%d/%m/%Y")
        time = datetime.now().strftime("%H:%M:%S")



        for i, openbox in enumerate(openboxes):
            for j, pizza in enumerate(pizzas):
                if self._is_inside_box(tuple(openbox), tuple(pizza)):
                    openbox_id = openbox_ids[i] 
                    pizza_id = pizza_ids[j]
                    
                    existing = self.tracked.get(openbox_id)
                    
                    # if this box isn't mapped yet and this pizza isn't already mapped:
                    if existing is None and pizza_id not in self.tracked.values():
                        self.tracked[openbox_id] = pizza_id
                        
                        self.database["Date"].append(date)
                        self.database["Time"].append(time)
                        self.database["Count"] += 1
                    
                    # if it's already mapped to the same pizza, continue
                    elif existing == pizza_id:
                        continue
                        
                    # otherwise there's a conflict (box different pizza or pizza different box), skip
                    else:
                        continue

    def _count_sale(self, results):
        """Counts the number of sales by processing YOLO detection results."""

        # Get bounding boxes and their corresponding id
        pizzas, openboxes, pizza_ids, openbox_ids = self._get_config(results)
        
        # Only process bounding boxes if both contain coordinates
        if pizzas and openboxes:
            self._process_bbox(pizzas, openboxes, pizza_ids, openbox_ids)
        
        return self.database["Count"]
        
    def __call__(self, results):
        return self._count_sale(results)

# 1. YOLO detect pizza and opened box.
# 2. function `get_config` store pizza and openbox coordinates
# 3. The `process_` function run and the function `is_inside_box` return True if any pizza lie inside the opened box.
# 4. if pizza box lie inside the opened box. Store the track id of both pizza and the box in a dictionary, then count that as a sale