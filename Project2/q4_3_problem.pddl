(define (problem pacmanP)
    (:domain pacmanD)

    (:objects 
             pac1 pac2 -pacman
             loc_x1y1 loc_x1y3 loc_x1y4 loc_x2y1 loc_x2y2 loc_x2y3 loc_x2y4 loc_x3y1 loc_x3y2 loc_x3y3 loc_x4y1 loc_x4y2 loc_x4y3 loc_x4y4 -loc
    )

    (:init (pacman-at pac1 loc_x4y4) (south loc_x2y1 loc_x1y1) 
           (south loc_x2y3 loc_x1y3) (east loc_x1y4 loc_x1y3) (south loc_x2y4 loc_x1y4) (west loc_x1y3 loc_x1y4)

           (south loc_x3y1 loc_x2y1) (east loc_x2y2 loc_x2y1) (north loc_x1y1 loc_x2y1) (west loc_x2y1 loc_x2y2) (east loc_x2y3 loc_x2y2) (south loc_x3y2 loc_x2y2) 
           (east loc_x2y4 loc_x2y3) (south loc_x3y3 loc_x2y3) (west loc_x2y2 loc_x2y3) (north loc_x1y3 loc_x2y3) 
           (west loc_x2y3 loc_x2y4) (north loc_x1y4 loc_x2y4)
           (pacman-at pac2 loc_x3y1)
           
           (south loc_x4y1 loc_x3y1) (east loc_x3y2 loc_x3y1) (north loc_x2y1 loc_x3y1) (west loc_x3y1 loc_x3y2) (east loc_x3y3 loc_x3y2) (south loc_x4y2 loc_x3y2) 
           (north loc_x2y2 loc_x3y2) (south loc_x4y3 loc_x3y3) (west loc_x3y2 loc_x3y3) (north loc_x2y3 loc_x3y3)
           
           (east loc_x4y2 loc_x4y1) (north loc_x3y1 loc_x4y1) (west loc_x4y1 loc_x4y2) (east loc_x4y3 loc_x4y2) (north loc_x3y2 loc_x4y2) (north loc_x3y3 loc_x4y3) 
           (east loc_x4y4 loc_x4y3) (west loc_x4y2 loc_x4y3)  (west loc_x4y3 loc_x4y4)
           
           (fruit-at loc_x1y1)                     (fruit-at loc_x1y3) (fruit-at loc_x1y4)
           (fruit-at loc_x2y1) (fruit-at loc_x2y2) (fruit-at loc_x2y3) (fruit-at loc_x2y4)
           (fruit-at loc_x3y1) (fruit-at loc_x3y2) (fruit-at loc_x3y3)
           (fruit-at loc_x4y1) (fruit-at loc_x4y2) (fruit-at loc_x4y3) (fruit-at loc_x4y4)
           )

    (:goal (and
           (not (fruit-at loc_x1y1))                           (not (fruit-at loc_x1y3)) (not (fruit-at loc_x1y4))
           (not (fruit-at loc_x2y1)) (not (fruit-at loc_x2y2)) (not (fruit-at loc_x2y3)) (not (fruit-at loc_x2y4))
           (not (fruit-at loc_x3y1)) (not (fruit-at loc_x3y2)) (not (fruit-at loc_x3y3))
           (not (fruit-at loc_x4y1)) (not (fruit-at loc_x4y2)) (not (fruit-at loc_x4y3)) (not (fruit-at loc_x4y4))
    )))
